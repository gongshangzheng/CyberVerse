"""
真实数字人视频生成集成测试。

前置条件（不满足则 skip）：
  - CUDA 可用
  - checkpoints/SoulX-FlashHead-1_3B 与 wav2vec 权重已下载
  - 已安装 flash_head 依赖（含 torch、opencv 等）
  - 系统 ``ffmpeg`` 在 PATH 中（用于 H.264 与 **AAC 音轨 mux**）
  - 存在 ``examples/girl.png`` 作为数字人条件图
  - 存在 ``examples/podcast_sichuan_16k.wav``（16kHz mono PCM）；每次测试**随机截取最多 60s** 片段喂入（插件 deque 仅保留**末尾 8s** 参与推理，长段可保证这 8s 为连续真实语音）

运行（仓库根目录）::

    pytest tests/integration/test_flash_head_generates_real_video.py -m integration -v -s

可选环境变量：

- ``CYBERVERSE_AVATAR_TEST_MP4``：输出 MP4 路径，默认 ``artifacts/flash_head_smoke.mp4``。
- ``CYBERVERSE_FLASH_HEAD_MIN_INIT_SEC``：若设置（如 ``10``），则要求 ``initialize`` 墙钟时间不低于该值，用于发现「秒过」的异常环境；默认不校验（避免慢盘/热缓存误杀）。
- ``CYBERVERSE_FLASH_HEAD_AUDIO_SEED``：若设置，则随机音频起点可复现（传入 ``random.Random.seed`` 的任意可哈希值）。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
# 集成测试用人像底图（FlashHead 内部会按 infer target_size 做 resize/center crop）
AVATAR_IMAGE = REPO_ROOT / "examples" / "girl.png"
EXAMPLE_WAV_16K = REPO_ROOT / "examples" / "podcast_sichuan_16k.wav"
SR_16K = 16000
# 每次随机截取的时长（秒）；若文件更短则使用整段
AUDIO_SEGMENT_SECONDS = 5
# FlashHead 插件 deque 长度，短于此时 skip（否则预填零占比过大）
MIN_WAV_FRAMES_FOR_TEST = SR_16K * 8


def _repo_paths_ok() -> bool:
    ckpt = REPO_ROOT / "checkpoints" / "SoulX-FlashHead-1_3B"
    w2v = REPO_ROOT / "checkpoints" / "wav2vec2-base-960h"
    return ckpt.is_dir() and any(ckpt.iterdir()) and w2v.is_dir() and any(w2v.iterdir())


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _load_wav_pcm16_mono_random_segment(
    path: Path,
    *,
    segment_seconds: float,
    rng: random.Random,
) -> tuple[bytes, int, int]:
    """从 WAV 中随机截取一段 contiguous PCM（16kHz mono int16）。

    返回 ``(pcm_bytes, start_frame, n_frames)``。若文件总长短于 ``segment_seconds``，
    则 ``n_frames`` 为整文件帧数且 ``start_frame == 0``。
    """
    want = int(SR_16K * segment_seconds)
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR_16K or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"expected 16kHz mono 16-bit PCM, got sr={w.getframerate()} "
                f"ch={w.getnchannels()} width={w.getsampwidth()}"
            )
        total = w.getnframes()
        if total < want:
            start = 0
            seg = total
        else:
            seg = want
            start = rng.randint(0, total - seg)
        w.setpos(start)
        data = w.readframes(seg)
    return data, start, seg


def _pcm_s16le_tail_for_duration(pcm: bytes, sample_rate: int, duration_sec: float) -> bytes:
    """取 PCM 末尾若干采样，使时长约等于 ``duration_sec``（与 deque 末尾语义一致）；不足则前补静音。"""
    need_samples = max(1, int(round(duration_sec * sample_rate)))
    need_bytes = need_samples * 2
    if len(pcm) >= need_bytes:
        return pcm[-need_bytes:]
    pad = need_bytes - len(pcm)
    return bytes(pad) + pcm


def _write_mp4_h264_ffmpeg(
    frames: np.ndarray,
    fps: int,
    path: Path,
    *,
    pcm_s16le_mono: bytes | None = None,
    audio_sample_rate: int = SR_16K,
) -> None:
    """libx264 + yuv420p；若提供 ``pcm_s16le_mono`` 则编码 AAC 音轨并 mux（需 ffmpeg）。"""
    t, h, w, c = frames.shape
    assert c == 3
    path.parent.mkdir(parents=True, exist_ok=True)
    video_in = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if pcm_s16le_mono is None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *video_in,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(path),
        ]
        proc = subprocess.run(
            cmd,
            input=frames.tobytes(),
            capture_output=True,
            timeout=300,
        )
    else:
        fd, pcm_path = tempfile.mkstemp(suffix=".pcm", prefix="cyberverse_flash_head_")
        os.close(fd)
        try:
            Path(pcm_path).write_bytes(pcm_s16le_mono)
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *video_in,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(audio_sample_rate),
                "-i",
                pcm_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(path),
            ]
            proc = subprocess.run(
                cmd,
                input=frames.tobytes(),
                capture_output=True,
                timeout=300,
            )
        finally:
            try:
                os.unlink(pcm_path)
            except OSError:
                pass
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 编码失败 (exit {proc.returncode}): {err}")


def _write_mp4_opencv_mp4v(frames: np.ndarray, fps: int, path: Path) -> None:
    """frames: (T, H, W, 3) uint8 RGB。部分播放器对 mp4v 支持差，仅作回退。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, c = frames.shape
    assert c == 3, f"expected RGB, got shape {frames.shape}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter could not open {path}")
    try:
        for i in range(t):
            bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _write_mp4_rgb(
    frames: np.ndarray,
    fps: int,
    path: Path,
    *,
    pcm_s16le_mono: bytes | None = None,
    audio_sample_rate: int = SR_16K,
) -> None:
    if pcm_s16le_mono is not None and not shutil.which("ffmpeg"):
        raise RuntimeError("合并音轨需要系统 PATH 中的 ffmpeg")
    if shutil.which("ffmpeg"):
        _write_mp4_h264_ffmpeg(
            frames,
            fps,
            path,
            pcm_s16le_mono=pcm_s16le_mono,
            audio_sample_rate=audio_sample_rate,
        )
    else:
        if pcm_s16le_mono:
            raise RuntimeError("OpenCV 路径无法写入音轨，请安装 ffmpeg")
        _write_mp4_opencv_mp4v(frames, fps, path)


def _assert_rgb_frames_not_static_placeholder(frames: np.ndarray) -> None:
    """在写文件前检查：非纯色、相邻帧有足够差异（避免静音+灰图类「假通过」）。"""
    assert frames.ndim == 4 and frames.shape[-1] == 3
    flat = frames.reshape(frames.shape[0], -1)
    spatial_std = flat.std(axis=1)
    assert float(spatial_std.min()) >= 6.0, (
        f"帧内对比度过低（疑似纯色/坏图），spatial_std min={float(spatial_std.min()):.4f}"
    )
    if frames.shape[0] >= 2:
        d = np.abs(frames[1:].astype(np.int32) - frames[:-1].astype(np.int32))
        mean_step = float(d.mean())
        max_step = float(d.max())
        # 单 chunk + 短视频里口型变化可以较缓，全零音频时均值可 <1；此处用「均值或峰值」避免误杀真实推理（实测约 2.98）
        assert mean_step >= 2.5 or max_step >= 24.0, (
            f"相邻帧几乎无变化（疑似静态或无效推理），mean_abs_diff={mean_step:.4f}, max={max_step:.1f}"
        )


def _ffprobe_assert_video_ok(
    path: Path, expect_frames: int, *, require_audio: bool = False
) -> None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_frames,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ffprobe 失败: {(proc.stderr or proc.stdout).strip()}"
    info = json.loads(proc.stdout)
    streams = info.get("streams") or []
    assert streams, "ffprobe 未解析到视频流"
    st = streams[0]
    assert st.get("codec_name") in ("h264", "mpeg4"), st
    nf = st.get("nb_frames")
    if nf is not None and str(nf).isdigit():
        got = int(nf)
        assert abs(got - expect_frames) <= 1, f"ffprobe nb_frames={got}, 期望约 {expect_frames}"
    if require_audio:
        ap = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert ap.returncode == 0, f"无音频轨: {(ap.stderr or '').strip()}"
        acodec = (ap.stdout or "").strip()
        assert acodec == "aac", f"期望 AAC 音轨，得到 codec={acodec!r}"
    # 强制完整解码一遍，避免「有 moov 但帧损坏」仍过 ffprobe 元数据
    dec = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert dec.returncode == 0, f"ffmpeg 解码验证失败: {(dec.stderr or '').strip()}"


@pytest.mark.asyncio
async def test_flash_head_generates_mp4_file():
    if not _cuda_available():
        pytest.skip("需要 CUDA 与 flash_head 环境以生成真实视频")
    if not _repo_paths_ok():
        pytest.skip("缺少 checkpoints/SoulX-FlashHead-1_3B 或 checkpoints/wav2vec2-base-960h")
    if not AVATAR_IMAGE.is_file():
        pytest.skip(f"缺少测试用人像图片: {AVATAR_IMAGE}")
    if not EXAMPLE_WAV_16K.is_file():
        pytest.skip(f"缺少测试用语音: {EXAMPLE_WAV_16K}")

    with wave.open(str(EXAMPLE_WAV_16K), "rb") as _w:
        if _w.getnframes() < MIN_WAV_FRAMES_FOR_TEST:
            pytest.skip(
                f"示例 WAV 短于 {MIN_WAV_FRAMES_FOR_TEST / SR_16K:.0f}s，无法稳定填满 FlashHead 8s 窗口"
            )

    from inference.core.config import load_config
    from inference.core.types import AudioChunk, PluginConfig
    from inference.plugins.avatar.flash_head_plugin import FlashHeadAvatarPlugin

    os.chdir(REPO_ROOT)

    raw = load_config(str(REPO_ROOT / "cyberverse_config.yaml"))
    avatar_cfg = raw["inference"]["avatar"]
    section = avatar_cfg["flash_head"]
    runtime = avatar_cfg.get("runtime", {})
    params = {k: v for k, v in {**runtime, **section}.items() if k != "plugin_class"}
    for key in ("checkpoint_dir", "wav2vec_dir", "models_dir"):
        if key in params and params[key]:
            p = Path(params[key])
            if not p.is_absolute():
                params[key] = str(REPO_ROOT / p)

    plugin = FlashHeadAvatarPlugin()
    try:
        t0 = time.perf_counter()
        await plugin.initialize(PluginConfig(plugin_name="avatar.flash_head", params=params))
        init_sec = time.perf_counter() - t0
        min_init = os.environ.get("CYBERVERSE_FLASH_HEAD_MIN_INIT_SEC")
        if min_init is not None and min_init.strip():
            need = float(min_init)
            assert init_sec >= need, (
                f"initialize 仅耗时 {init_sec:.2f}s < {need}s，疑似未真实加载权重（可检查是否走错/缓存异常）"
            )

        # 使用仓库内示例人像；pipeline.prepare_params 会按 target_size 处理尺寸
        await plugin.set_avatar(str(AVATAR_IMAGE), use_face_crop=False)

        rng = random.Random()
        seed_env = os.environ.get("CYBERVERSE_FLASH_HEAD_AUDIO_SEED")
        if seed_env is not None and seed_env.strip():
            rng.seed(seed_env.strip())
        pcm, wav_start, wav_nframes = _load_wav_pcm16_mono_random_segment(
            EXAMPLE_WAV_16K,
            segment_seconds=AUDIO_SEGMENT_SECONDS,
            rng=rng,
        )

        async def one_chunk_stream():
            yield AudioChunk(
                data=pcm,
                sample_rate=16000,
                channels=1,
                format="pcm_s16le",
                is_final=True,
            )

        video_chunks: list = []
        infer_started = time.perf_counter()
        first_chunk_sec: float | None = None
        async for vc in plugin.generate_stream(one_chunk_stream()):
            if vc is not None:
                if first_chunk_sec is None:
                    first_chunk_sec = time.perf_counter() - infer_started
                video_chunks.append(vc)

        assert video_chunks, "FlashHead 未产出任何 VideoChunk，请检查 GPU/权重/日志"
        frames = np.concatenate([c.frames for c in video_chunks], axis=0)
        assert frames.ndim == 4 and frames.shape[-1] == 3
        assert frames.dtype == np.uint8

        _assert_rgb_frames_not_static_placeholder(frames)

        out = Path(
            os.environ.get(
                "CYBERVERSE_AVATAR_TEST_MP4",
                str(REPO_ROOT / "artifacts" / "flash_head_smoke.mp4"),
            )
        )
        fps = int(video_chunks[0].fps) or 25
        if not shutil.which("ffmpeg"):
            pytest.skip("输出含音轨的 MP4 需要 ffmpeg（PATH）")
        video_dur = frames.shape[0] / float(fps)
        pcm_mux = _pcm_s16le_tail_for_duration(pcm, SR_16K, video_dur)
        _write_mp4_rgb(
            frames,
            fps,
            out,
            pcm_s16le_mono=pcm_mux,
            audio_sample_rate=SR_16K,
        )

        size = out.stat().st_size
        assert frames.shape[0] >= 8, f"产出帧数过少: {frames.shape[0]}"
        assert size > 2_000, f"生成的 MP4 异常小 ({size} bytes): {out}"
        _ffprobe_assert_video_ok(
            out, expect_frames=int(frames.shape[0]), require_audio=True
        )

        enc = "libx264+aac(ffmpeg)"
        dur_s = wav_nframes / SR_16K
        print(
            f"\n[integration] init_wall={init_sec:.2f}s, first_chunk_wall={first_chunk_sec:.2f}s, "
            f"编码={enc}"
        )
        print(
            f"[integration] 音频随机片: start_frame={wav_start}, n_frames={wav_nframes} "
            f"({dur_s:.2f}s)，推理使用 deque 末尾 8s"
        )
        print(f"[integration] 已生成数字人视频: {out.resolve()} ({size} bytes, {frames.shape[0]} 帧)")
    finally:
        await plugin.shutdown()
