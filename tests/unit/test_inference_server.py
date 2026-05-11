from unittest.mock import MagicMock, patch

from inference.server import InferenceServer


def _make_server(config: dict) -> InferenceServer:
    with patch("inference.server.load_config", return_value=config):
        with patch("inference.server.grpc.aio.server", return_value=MagicMock()):
            return InferenceServer("cyberverse_config.yaml")


def test_build_plugin_config_passes_root_warmup_to_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.flash_head",
        {
            "plugin_class": "pkg.Plugin",
            "device": "cuda:0",
            "compile_model": True,
            "compile_vae": True,
            "dist_worker_main_thread": True,
            "infer_params": {
                "frame_num": 33,
                "motion_frames_latent_num": 2,
                "tgt_fps": 25,
                "sample_rate": 16000,
                "sample_shift": 5,
                "color_correction_strength": 1.0,
                "cached_audio_duration": 8,
                "num_heads": 12,
                "height": 512,
                "width": 512,
            },
        },
    )

    assert plugin_config.plugin_name == "avatar.flash_head"
    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 2,
        "device": "cuda:0",
        "compile_model": True,
        "compile_vae": True,
        "dist_worker_main_thread": True,
        "infer_params": {
            "frame_num": 33,
            "motion_frames_latent_num": 2,
            "tgt_fps": 25,
            "sample_rate": 16000,
            "sample_shift": 5,
            "color_correction_strength": 1.0,
            "cached_audio_duration": 8,
            "num_heads": 12,
            "height": 512,
            "width": 512,
        },
    }
    assert plugin_config.shared["warmup"] == config["warmup"]


def test_build_plugin_config_model_values_override_avatar_runtime_defaults():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.live_act",
        {
            "plugin_class": "pkg.Plugin",
            "world_size": 1,
            "compile_wan_model": False,
            "compile_vae_decode": False,
            "dist_worker_main_thread": True,
            "infer_params": {
                "size": "320*480",
                "fps": 20,
                "audio_cfg": 1.0,
            },
        },
    )

    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 1,
        "compile_wan_model": False,
        "compile_vae_decode": False,
        "dist_worker_main_thread": True,
        "infer_params": {
            "size": "320*480",
            "fps": 20,
            "audio_cfg": 1.0,
        },
    }


def test_build_plugin_config_does_not_pass_root_warmup_to_non_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "llm",
        "llm.openai",
        {
            "plugin_class": "pkg.Plugin",
            "model": "gpt-4o",
        },
    )

    assert plugin_config.plugin_name == "llm.openai"
    assert plugin_config.params == {"model": "gpt-4o"}
    assert plugin_config.shared == {}


def test_build_plugin_config_passes_omni_models_to_persona_plugins():
    config = {
        "inference": {
            "omni": {
                "qwen_omni": {
                    "plugin_class": "pkg.Qwen",
                    "model": "qwen3.5-omni-flash-realtime",
                }
            },
            "persona": {
                "persona": {
                    "plugin_class": "pkg.Persona",
                    "model_provider": "qwen_omni",
                }
            },
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "persona",
        "persona.persona",
        config["inference"]["persona"]["persona"],
    )

    assert plugin_config.plugin_name == "persona.persona"
    assert plugin_config.params == {"model_provider": "qwen_omni"}
    assert plugin_config.shared["omni"] == config["inference"]["omni"]


def test_agent_worker_settings_default_to_embedded_endpoint():
    server = _make_server({"inference": {}})

    assert server._agent_worker_settings() == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8090,
    }


def test_agent_worker_settings_can_be_disabled():
    server = _make_server(
        {
            "inference": {
                "agent_worker": {
                    "enabled": False,
                    "host": "0.0.0.0",
                    "port": 9009,
                }
            }
        }
    )

    assert server._agent_worker_settings() == {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 9009,
    }
