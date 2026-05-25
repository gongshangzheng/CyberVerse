import { ref, watch } from 'vue'
import type {
  ConnectionState,
  AVSyncDebugState,
  AVSegmentTimeline,
  FrameJitterStats,
  WebRTCNetworkStats,
} from './useWebRTC'
import {
  createAVPlayoutEstimatorState,
  estimateAVPlayoutFromStats,
  formatJitterBufferDelta,
} from './useWebRTC'

// ─── Diagnostics ──────────────────────────────────────────────────────────
const T0 = performance.now()
function ts() {
  return `+${((performance.now() - T0) / 1000).toFixed(3)}s`
}

// Frame tracking state (module-level, same pattern as useWebRTC)
let videoFrameCount = 0
let videoFirstFrameTime: number | null = null
let videoLastFrameWallMs: number | null = null
let audioPlayWallMs: number | null = null
let debugPollTimer: ReturnType<typeof setInterval> | null = null

// Jitter measurement
const JITTER_WINDOW = 120
let frameArrivalTimes: number[] = []
const AV_SEGMENT_TIMELINE_LIMIT = 80
const AV_SEGMENT_MATCH_TOLERANCE_MS = 80
const AV_MARKER_MATCH_TOLERANCE_MS = 120
let avSegmentTimelines: AVSegmentTimeline[] = []
let avSegmentMediaBaseTurnSeq: number | null = null
let avSegmentMediaBaseMs: number | null = null
let avSegmentPresentationLoggingEnabled = false
let lastPresentedSegmentKey = ''
let avCalibrationEnabled = false
let avCalibrationCanvas: HTMLCanvasElement | null = null
let avCalibrationCtx: CanvasRenderingContext2D | null = null
let avCalibrationVideoHot = false
let avCalibrationAudioContext: AudioContext | null = null
let avCalibrationAudioSource: MediaStreamAudioSourceNode | null = null
let avCalibrationWorklet: AudioWorkletNode | null = null
let avCalibrationSilentGain: GainNode | null = null
let pendingAVCalibrationAudioOutputTimes: number[] = []
let pendingAVCalibrationVideoEvents: Array<{ mediaInTurnMs: number; compositorTimeMs: number }> = []

type AVCalibrationMarker = {
  id: number
  turnSeq: number
  mediaTimeMs: number
  frequencyHz: number
  durationMs: number
  videoCompositorTimeMs: number | null
  audioOutputTimeMs: number | null
  logged: boolean
}

const avCalibrationMarkers = new Map<number, AVCalibrationMarker>()

function resetJitterState() {
  frameArrivalTimes = []
}

function recordFrameArrival(wallMs: number) {
  frameArrivalTimes.push(wallMs)
  if (frameArrivalTimes.length > JITTER_WINDOW) {
    frameArrivalTimes.shift()
  }
}

function resetAVSegmentDiagnostics() {
  avSegmentTimelines = []
  avSegmentMediaBaseTurnSeq = null
  avSegmentMediaBaseMs = null
  lastPresentedSegmentKey = ''
  avCalibrationVideoHot = false
  pendingAVCalibrationAudioOutputTimes = []
  pendingAVCalibrationVideoEvents = []
  avCalibrationMarkers.clear()
}

function segmentKey(seg: AVSegmentTimeline): string {
  return `${seg.turnSeq}:${seg.segmentSeq}`
}

function numberField(data: any, key: string): number {
  const value = Number(data?.[key])
  return Number.isFinite(value) ? value : 0
}

function parseAVSegmentTimeline(data: any): AVSegmentTimeline | null {
  const turnSeq = numberField(data, 'turn_seq')
  const segmentSeq = numberField(data, 'segment_seq')
  if (turnSeq <= 0 || segmentSeq <= 0) {
    return null
  }
  return {
    turnSeq,
    segmentSeq,
    mediaStartMs: numberField(data, 'media_start_ms'),
    durationMs: numberField(data, 'duration_ms'),
    videoFrames: numberField(data, 'video_frames'),
    fps: numberField(data, 'fps'),
    audioSamples: numberField(data, 'audio_samples'),
    sampleRate: numberField(data, 'sample_rate'),
    publishWallMs: numberField(data, 'publish_wall_ms'),
    receivedWallMs: Date.now(),
    markerId: numberField(data, 'marker_id'),
    markerMediaTimeMs: numberField(data, 'marker_media_time_ms'),
    markerDurationMs: numberField(data, 'marker_duration_ms'),
    markerFrequencyHz: numberField(data, 'marker_frequency_hz'),
  }
}

function appendAVSegmentTimeline(seg: AVSegmentTimeline) {
  avSegmentTimelines = avSegmentTimelines
    .filter(existing => segmentKey(existing) !== segmentKey(seg))
    .concat(seg)
    .sort((a, b) => a.turnSeq === b.turnSeq ? a.segmentSeq - b.segmentSeq : a.turnSeq - b.turnSeq)
    .slice(-AV_SEGMENT_TIMELINE_LIMIT)
  if (seg.markerId > 0 && seg.markerMediaTimeMs > 0) {
    avCalibrationMarkers.set(seg.markerId, {
      id: seg.markerId,
      turnSeq: seg.turnSeq,
      mediaTimeMs: seg.markerMediaTimeMs,
      frequencyHz: seg.markerFrequencyHz,
      durationMs: seg.markerDurationMs,
      videoCompositorTimeMs: null,
      audioOutputTimeMs: null,
      logged: false,
    })
    attachPendingAVCalibrationVideoEvents()
    attachPendingAVCalibrationAudioMarkers()
    pruneAVCalibrationMarkers()
  }
}

function latestTurnSegments(): AVSegmentTimeline[] {
  const latest = avSegmentTimelines[avSegmentTimelines.length - 1]
  if (!latest) return []
  return avSegmentTimelines.filter(seg => seg.turnSeq === latest.turnSeq)
}

function pruneAVCalibrationMarkers() {
  const markers = [...avCalibrationMarkers.values()].sort((a, b) => a.mediaTimeMs - b.mediaTimeMs)
  for (const marker of markers.slice(0, Math.max(0, markers.length - 40))) {
    avCalibrationMarkers.delete(marker.id)
  }
}

function maybeLogAVCalibrationMarker(marker: AVCalibrationMarker) {
  if (marker.logged || marker.videoCompositorTimeMs === null || marker.audioOutputTimeMs === null) return
  marker.logged = true
  const perceivedAVOffsetMs = marker.videoCompositorTimeMs - marker.audioOutputTimeMs
  console.log(
    `[DirectRTC][${ts()}] AV calibration marker=${marker.id}` +
      ` perceivedAVOffsetMs=${perceivedAVOffsetMs.toFixed(1)}` +
      ` videoCompositor=${marker.videoCompositorTimeMs.toFixed(1)}ms` +
      ` audioOutput=${marker.audioOutputTimeMs.toFixed(1)}ms` +
      ` markerMedia=${marker.mediaTimeMs}ms freq=${marker.frequencyHz}Hz`
  )
}

function findCalibrationMarkerByMediaTime(mediaInTurnMs: number): AVCalibrationMarker | null {
  let best: AVCalibrationMarker | null = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const marker of avCalibrationMarkers.values()) {
    if (marker.videoCompositorTimeMs !== null) continue
    const distance = Math.abs(marker.mediaTimeMs - mediaInTurnMs)
    if (distance < bestDistance) {
      best = marker
      bestDistance = distance
    }
  }
  return best && bestDistance <= AV_MARKER_MATCH_TOLERANCE_MS ? best : null
}

function nextCalibrationMarkerForAudio(): AVCalibrationMarker | null {
  return [...avCalibrationMarkers.values()]
    .filter(marker => marker.audioOutputTimeMs === null)
    .sort((a, b) => a.mediaTimeMs - b.mediaTimeMs)[0] ?? null
}

function attachPendingAVCalibrationVideoEvents() {
  const remaining: Array<{ mediaInTurnMs: number; compositorTimeMs: number }> = []
  for (const event of pendingAVCalibrationVideoEvents) {
    const marker = findCalibrationMarkerByMediaTime(event.mediaInTurnMs)
    if (!marker) {
      remaining.push(event)
      continue
    }
    marker.videoCompositorTimeMs = event.compositorTimeMs
    maybeLogAVCalibrationMarker(marker)
  }
  pendingAVCalibrationVideoEvents = remaining.slice(-10)
}

function attachPendingAVCalibrationAudioMarkers() {
  while (pendingAVCalibrationAudioOutputTimes.length > 0) {
    const marker = nextCalibrationMarkerForAudio()
    if (!marker) return
    marker.audioOutputTimeMs = pendingAVCalibrationAudioOutputTimes.shift() ?? null
    maybeLogAVCalibrationMarker(marker)
  }
}

function detectCalibrationFlash(el: HTMLVideoElement): boolean {
  if (!avCalibrationEnabled || el.videoWidth <= 0 || el.videoHeight <= 0) return false
  if (!avCalibrationCanvas) {
    avCalibrationCanvas = document.createElement('canvas')
    avCalibrationCanvas.width = 24
    avCalibrationCanvas.height = 24
    avCalibrationCtx = avCalibrationCanvas.getContext('2d', { willReadFrequently: true })
  }
  const ctx = avCalibrationCtx
  if (!ctx) return false

  const sourceW = Math.min(el.videoWidth, Math.max(48, Math.floor(el.videoWidth / 5)))
  const sourceH = Math.min(el.videoHeight, Math.max(48, Math.floor(el.videoHeight / 5)))
  try {
    ctx.drawImage(el, 0, 0, sourceW, sourceH, 0, 0, avCalibrationCanvas.width, avCalibrationCanvas.height)
    const pixels = ctx.getImageData(0, 0, avCalibrationCanvas.width, avCalibrationCanvas.height).data
    let hot = 0
    const total = pixels.length / 4
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 180 && pixels[i + 1] < 100 && pixels[i + 2] > 180) {
        hot++
      }
    }
    return hot / total > 0.35
  } catch {
    return false
  }
}

function recordAVCalibrationVideoFrame(
  el: HTMLVideoElement,
  rvfcNowMs: DOMHighResTimeStamp,
  meta: VideoFrameMeta,
  mediaInTurnMs: number,
) {
  if (!avCalibrationEnabled) return
  const hot = detectCalibrationFlash(el)
  if (!hot) {
    avCalibrationVideoHot = false
    return
  }
  if (avCalibrationVideoHot) return
  avCalibrationVideoHot = true

  const marker = findCalibrationMarkerByMediaTime(mediaInTurnMs)
  const compositorTimeMs = Number.isFinite(meta.expectedDisplayTime ?? NaN)
    ? Number(meta.expectedDisplayTime)
    : rvfcNowMs
  if (!marker) {
    pendingAVCalibrationVideoEvents.push({ mediaInTurnMs, compositorTimeMs })
    pendingAVCalibrationVideoEvents = pendingAVCalibrationVideoEvents.slice(-10)
    return
  }
  marker.videoCompositorTimeMs = compositorTimeMs
  maybeLogAVCalibrationMarker(marker)
}

function recordAVSegmentPresentation(
  el: HTMLVideoElement,
  rvfcNowMs: DOMHighResTimeStamp,
  mediaTimeSeconds: number,
  presentedFrames: number,
  expectedDisplayTime?: DOMHighResTimeStamp,
) {
  if (!avSegmentPresentationLoggingEnabled && !avCalibrationEnabled) return
  const segments = latestTurnSegments()
  if (segments.length === 0) return

  const turnSeq = segments[0].turnSeq
  const mediaTimeMs = mediaTimeSeconds * 1000
  if (avSegmentMediaBaseTurnSeq !== turnSeq || avSegmentMediaBaseMs === null) {
    avSegmentMediaBaseTurnSeq = turnSeq
    avSegmentMediaBaseMs = mediaTimeMs - segments[0].mediaStartMs
    lastPresentedSegmentKey = ''
  }

  const mediaInTurnMs = mediaTimeMs - avSegmentMediaBaseMs
  recordAVCalibrationVideoFrame(el, rvfcNowMs, { mediaTime: mediaTimeSeconds, presentedFrames, expectedDisplayTime }, mediaInTurnMs)
  if (!avSegmentPresentationLoggingEnabled) return

  const seg = segments.find(item =>
    mediaInTurnMs >= item.mediaStartMs - AV_SEGMENT_MATCH_TOLERANCE_MS &&
    mediaInTurnMs < item.mediaStartMs + item.durationMs + AV_SEGMENT_MATCH_TOLERANCE_MS
  )
  if (!seg) return

  const key = segmentKey(seg)
  if (key === lastPresentedSegmentKey) return
  lastPresentedSegmentKey = key
  console.log(
    `[DirectRTC][${ts()}] AV segment presented turn=${seg.turnSeq} segment=${seg.segmentSeq}` +
      ` mediaInTurn=${Math.round(mediaInTurnMs)}ms expected=${seg.mediaStartMs}-${seg.mediaStartMs + seg.durationMs}ms` +
      ` rvfcMedia=${mediaTimeSeconds.toFixed(3)}s presentedFrames=${presentedFrames}`
  )
}

const avCalibrationWorkletSource = `
class AVMarkerDetector extends AudioWorkletProcessor {
  constructor() {
    super()
    this.previousHot = false
    this.lastPostFrame = -Infinity
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (!channel || channel.length === 0) return true

    let energy = 0
    let sin = 0
    let cos = 0
    const freq = 1800
    const step = 2 * Math.PI * freq / sampleRate
    for (let i = 0; i < channel.length; i++) {
      const sample = channel[i]
      energy += sample * sample
      const angle = (currentFrame + i) * step
      sin += sample * Math.sin(angle)
      cos += sample * Math.cos(angle)
    }

    const rms = Math.sqrt(energy / channel.length)
    const tone = Math.sqrt(sin * sin + cos * cos) / channel.length
    const hot = rms > 0.10 && tone > 0.055 && tone / Math.max(rms, 0.0001) > 0.35
    if (hot && !this.previousHot && currentFrame - this.lastPostFrame > sampleRate * 0.3) {
      this.lastPostFrame = currentFrame
      this.port.postMessage({ contextTime: currentTime })
    }
    this.previousHot = hot
    return true
  }
}

registerProcessor('av-marker-detector', AVMarkerDetector)
`

function recordAVCalibrationAudioMarker(contextTime: number) {
  const ctx = avCalibrationAudioContext
  const timestamp = ctx && typeof ctx.getOutputTimestamp === 'function'
    ? ctx.getOutputTimestamp()
    : null
  const performanceTime = Number(timestamp?.performanceTime)
  const outputContextTime = Number(timestamp?.contextTime)
  const audioOutputTimeMs = Number.isFinite(performanceTime) && Number.isFinite(outputContextTime)
    ? performanceTime + (contextTime - outputContextTime) * 1000
    : performance.now()

  const marker = nextCalibrationMarkerForAudio()
  if (!marker) {
    pendingAVCalibrationAudioOutputTimes.push(audioOutputTimeMs)
    pendingAVCalibrationAudioOutputTimes = pendingAVCalibrationAudioOutputTimes.slice(-10)
    return
  }
  marker.audioOutputTimeMs = audioOutputTimeMs
  maybeLogAVCalibrationMarker(marker)
}

function stopAVCalibrationAudioMonitor() {
  avCalibrationAudioSource?.disconnect()
  avCalibrationWorklet?.disconnect()
  avCalibrationSilentGain?.disconnect()
  avCalibrationAudioSource = null
  avCalibrationWorklet = null
  avCalibrationSilentGain = null
  if (avCalibrationAudioContext && avCalibrationAudioContext.state !== 'closed') {
    void avCalibrationAudioContext.close()
  }
  avCalibrationAudioContext = null
}

async function startAVCalibrationAudioMonitor(track: MediaStreamTrack) {
  if (!avCalibrationEnabled || avCalibrationWorklet) return
  const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext
  if (!AudioContextCtor) return

  try {
    const ctx = new AudioContextCtor() as AudioContext
    if (!ctx.audioWorklet || typeof AudioWorkletNode === 'undefined') {
      void ctx.close()
      return
    }
    const blob = new Blob([avCalibrationWorkletSource], { type: 'text/javascript' })
    const url = URL.createObjectURL(blob)
    try {
      await ctx.audioWorklet.addModule(url)
    } finally {
      URL.revokeObjectURL(url)
    }

    const source = ctx.createMediaStreamSource(new MediaStream([track]))
    const worklet = new AudioWorkletNode(ctx, 'av-marker-detector')
    const gain = ctx.createGain()
    gain.gain.value = 0
    worklet.port.onmessage = (event) => {
      const contextTime = Number(event.data?.contextTime)
      if (Number.isFinite(contextTime)) {
        recordAVCalibrationAudioMarker(contextTime)
      }
    }
    source.connect(worklet)
    worklet.connect(gain)
    gain.connect(ctx.destination)
    avCalibrationAudioContext = ctx
    avCalibrationAudioSource = source
    avCalibrationWorklet = worklet
    avCalibrationSilentGain = gain
    console.log(`[DirectRTC][${ts()}] AV calibration audio monitor enabled`)
  } catch (err) {
    stopAVCalibrationAudioMonitor()
    console.warn('[DirectRTC] AV calibration audio monitor failed:', err)
  }
}

function computeJitterStats(): FrameJitterStats {
  const times = frameArrivalTimes
  if (times.length < 2) {
    return { meanIntervalMs: 0, stddevMs: 0, maxIntervalMs: 0, p95IntervalMs: 0, stutterCount: 0, windowSize: 0 }
  }
  const intervals: number[] = []
  for (let i = 1; i < times.length; i++) {
    intervals.push(times[i] - times[i - 1])
  }
  intervals.sort((a, b) => a - b)
  const mean = intervals.reduce((s, v) => s + v, 0) / intervals.length
  const variance = intervals.reduce((s, v) => s + (v - mean) ** 2, 0) / intervals.length
  const stddev = Math.sqrt(variance)
  const p95 = intervals[Math.floor(intervals.length * 0.95)]
  const maxInterval = intervals[intervals.length - 1]
  const stutterThreshold = Math.max(mean * 2, 60)
  const stutterCount = intervals.filter(v => v > stutterThreshold).length
  return {
    meanIntervalMs: Math.round(mean * 10) / 10,
    stddevMs: Math.round(stddev * 10) / 10,
    maxIntervalMs: Math.round(maxInterval),
    p95IntervalMs: Math.round(p95),
    stutterCount,
    windowSize: intervals.length,
  }
}

const emptyNetworkStats = (): WebRTCNetworkStats => ({
  roundTripTimeMs: null,
  jitterMs: null,
  packetsLost: 0,
  packetsReceived: 0,
  lossRate: 0,
  bytesReceived: 0,
  framesDecoded: 0,
  framesDropped: 0,
  frameWidth: 0,
  frameHeight: 0,
  nackCount: 0,
  pliCount: 0,
  firCount: 0,
  jitterBufferDelayMs: null,
  jitterBufferEmittedCount: 0,
  codec: '',
})

const emptyDebugState = (): AVSyncDebugState => ({
  sessionId: '',
  connectionState: 'disconnected',
  audioSubscribedAtMs: null,
  videoSubscribedAtMs: null,
  audioUnmutedAtMs: null,
  videoUnmutedAtMs: null,
  firstPlayAtMs: null,
  videoFirstFrameAtMs: null,
  lastVideoFrameAtMs: null,
  fps: 0,
  videoCurrentTime: 0,
  readyState: 0,
  playbackRate: 1,
  decodedFrames: 0,
  droppedFrames: 0,
  totalFrames: 0,
  notes: [],
  jitter: { meanIntervalMs: 0, stddevMs: 0, maxIntervalMs: 0, p95IntervalMs: 0, stutterCount: 0, windowSize: 0 },
  network: null,
  avSync: null,
  segmentTimeline: null,
})

function resetState() {
  videoFrameCount = 0
  videoFirstFrameTime = null
  videoLastFrameWallMs = null
  audioPlayWallMs = null
  resetJitterState()
}

type VideoFrameMeta = { mediaTime: number; presentedFrames: number; expectedDisplayTime?: DOMHighResTimeStamp }
type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback: (cb: (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => void) => void
}

function attachVideoFrameCallback(el: HTMLVideoElement) {
  if (!('requestVideoFrameCallback' in el)) {
    (el as HTMLVideoElement).addEventListener('timeupdate', () => {
      if ((el as HTMLVideoElement).currentTime > 0) {
        videoLastFrameWallMs = Date.now()
      }
      if (videoFirstFrameTime === null && (el as HTMLVideoElement).currentTime > 0) {
        videoFirstFrameTime = performance.now()
      }
    })
    return
  }

  const rvfc = (el as VideoWithRVFC).requestVideoFrameCallback.bind(el)
  const onFrame = (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => {
    videoFrameCount++
    videoLastFrameWallMs = Date.now()
    recordFrameArrival(now)
    recordAVSegmentPresentation(el, now, meta.mediaTime, meta.presentedFrames, meta.expectedDisplayTime)

    if (videoFirstFrameTime === null) {
      videoFirstFrameTime = now
      console.log(
        `[DirectRTC][${ts()}] VIDEO first frame: mediaTime=${meta.mediaTime.toFixed(3)}s` +
          ` presentedFrames=${meta.presentedFrames}`
      )
    }

    rvfc(onFrame)
  }
  rvfc(onFrame)
}

// ──────────────────────────────────────────────────────────────────────────────

export function useDirectWebRTC() {
  const videoElement = ref<HTMLVideoElement | null>(null)
  const connectionState = ref<ConnectionState>('disconnected')
  const error = ref<string>('')
  const debugState = ref<AVSyncDebugState>(emptyDebugState())
  const needsPlaybackGesture = ref(false)
  const isOutputMuted = ref(false)

  let pc: RTCPeerConnection | null = null
  let sendSignaling: ((msg: any) => void) | null = null
  let localStream: MediaStream | null = null
  let remoteAudioTrackForCalibration: MediaStreamTrack | null = null
  let remoteAudioElement: HTMLAudioElement | null = null
  let dedicatedAudioOutput = false
  let networkStatsTimer: ReturnType<typeof setInterval> | null = null
  let lastAVSyncLogAtMs = 0
  let avSyncLoggingEnabled = false
  let avPlayoutEstimator = createAVPlayoutEstimatorState()

  function setAVSyncLoggingEnabled(enabled: boolean) {
    if (avSyncLoggingEnabled === enabled) return
    avSyncLoggingEnabled = enabled
    avSegmentPresentationLoggingEnabled = enabled
    avPlayoutEstimator = createAVPlayoutEstimatorState()
    lastAVSyncLogAtMs = 0
    resetAVSegmentDiagnostics()
    if (!enabled) {
      debugState.value.avSync = null
      debugState.value.segmentTimeline = null
    }
  }

  function configureAVCalibration(enabled: boolean) {
    avCalibrationEnabled = enabled
    if (!enabled) {
      stopAVCalibrationAudioMonitor()
      return
    }
    if (remoteAudioTrackForCalibration) {
      void startAVCalibrationAudioMonitor(remoteAudioTrackForCalibration)
    }
  }

  function handleAVSegmentDiagnostic(data: any) {
    const seg = parseAVSegmentTimeline(data)
    if (!seg) return
    appendAVSegmentTimeline(seg)
    debugState.value.segmentTimeline = seg
    if (!avSyncLoggingEnabled) return

    const videoDurationMs = seg.fps > 0 ? (seg.videoFrames * 1000) / seg.fps : 0
    const audioDurationMs = seg.sampleRate > 0 ? (seg.audioSamples * 1000) / seg.sampleRate : 0
    console.log(
      `[DirectRTC][${ts()}] AV segment timeline turn=${seg.turnSeq} segment=${seg.segmentSeq}` +
        ` mediaStart=${seg.mediaStartMs}ms duration=${seg.durationMs}ms` +
        ` video=${seg.videoFrames}f@${seg.fps}fps(${videoDurationMs.toFixed(1)}ms)` +
        ` audio=${seg.audioSamples}@${seg.sampleRate}Hz(${audioDurationMs.toFixed(1)}ms)` +
        ` publishWall=${seg.publishWallMs}` +
        (seg.markerId > 0 ? ` marker=${seg.markerId}@${seg.markerMediaTimeMs}ms` : '')
    )
  }

  // Serialize signaling: queue operations so addIceCandidate waits for setRemoteDescription
  let signalingChain: Promise<void> = Promise.resolve()

  // TURN ICE servers received from server via webrtc_config
  let pendingIceServers: RTCIceServer[] | null = null

  const isMuted = ref(false)
  let sentWebrtcReady = false

  const MIC_LEVEL_BARS = 16
  const micBarLevels = ref<number[]>(Array.from({ length: MIC_LEVEL_BARS }, () => 0))

  let micAudioContext: AudioContext | null = null
  let micAnalyser: AnalyserNode | null = null
  let micRafId = 0
  let micMediaSource: MediaStreamAudioSourceNode | null = null
  let micFreqBuffer: Uint8Array<ArrayBuffer> | null = null

  function stopMicMetering() {
    if (micRafId) {
      cancelAnimationFrame(micRafId)
      micRafId = 0
    }
    micMediaSource?.disconnect()
    micMediaSource = null
    micAnalyser?.disconnect()
    micAnalyser = null
    micFreqBuffer = null
    if (micAudioContext && micAudioContext.state !== 'closed') {
      void micAudioContext.close()
    }
    micAudioContext = null
    micBarLevels.value = Array.from({ length: MIC_LEVEL_BARS }, () => 0)
  }

  function attachMicMeter(mediaTrack: MediaStreamTrack) {
    stopMicMetering()
    if (mediaTrack.readyState !== 'live') {
      mediaTrack.addEventListener('unmute', () => attachMicMeter(mediaTrack), { once: true })
      return
    }

    try {
      const ctx = new AudioContext()
      micAudioContext = ctx
      const src = ctx.createMediaStreamSource(new MediaStream([mediaTrack]))
      micMediaSource = src
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.65
      micAnalyser = analyser
      micFreqBuffer = new Uint8Array(analyser.frequencyBinCount)
      src.connect(analyser)

      const tick = () => {
        if (!micAnalyser || !micFreqBuffer) return
        micAnalyser.getByteFrequencyData(micFreqBuffer)
        const data = micFreqBuffer
        const n = data.length
        const start = 1
        const usable = Math.max(1, n - start)
        const binW = usable / MIC_LEVEL_BARS
        const next: number[] = []
        for (let i = 0; i < MIC_LEVEL_BARS; i++) {
          const lo = Math.floor(start + i * binW)
          const hi = Math.floor(start + (i + 1) * binW)
          let sum = 0
          for (let j = lo; j < hi; j++) {
            sum += data[j] ?? 0
          }
          const bins = Math.max(1, hi - lo)
          const avg = sum / bins / 255
          next.push(Math.min(1, avg ** 0.65 * 3.2))
        }
        micBarLevels.value = next
        micRafId = requestAnimationFrame(tick)
      }

      void ctx.resume().then(() => {
        micRafId = requestAnimationFrame(tick)
      })
    } catch (e) {
      console.warn('[DirectRTC] mic meter failed', e)
      stopMicMetering()
    }
  }

  async function requestMicrophone(reason: 'connect' | 'click') {
    if (!window.isSecureContext) {
      throw new Error(`Microphone requires HTTPS or localhost (origin=${window.location.origin})`)
    }
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true })
      for (const track of localStream.getAudioTracks()) {
        // default to unmuted when acquiring
        track.enabled = true
        attachMicMeter(track)
      }
      isMuted.value = false
      pushNote(`mic acquired (${reason})`)
    } catch (e: any) {
      let userMsg: string = e?.message || String(e)
      if (e?.name === 'NotAllowedError') {
        userMsg =
          'Microphone access denied. ' +
          'On macOS: System Settings → Privacy & Security → Microphone → enable your browser, then click the mic button to retry. ' +
          'On other platforms: check browser site permissions and allow microphone for this page.'
        throw new Error(userMsg)
      }
      throw new Error(userMsg)
    }
  }

  function pushNote(note: string) {
    const next = [...debugState.value.notes, `${new Date().toISOString()} ${note}`]
    debugState.value.notes = next.slice(-10)
  }

  function remoteAudioEl(): HTMLAudioElement {
    if (!remoteAudioElement) {
      remoteAudioElement = new Audio()
      remoteAudioElement.autoplay = true
      remoteAudioElement.muted = isOutputMuted.value
      remoteAudioElement.volume = 1
      remoteAudioElement.addEventListener('play', () => {
        if (audioPlayWallMs === null) {
          audioPlayWallMs = performance.now()
          debugState.value.firstPlayAtMs = Date.now()
          pushNote('audio element playing')
        }
      })
    }
    return remoteAudioElement
  }

  function applyOutputMuted() {
    const muted = isOutputMuted.value
    const videoEl = videoElement.value
    if (videoEl) {
      videoEl.muted = muted
      videoEl.volume = muted ? 0 : 1
    }
    if (remoteAudioElement) {
      remoteAudioElement.muted = muted
      remoteAudioElement.volume = muted ? 0 : 1
    }
  }

  function clearRemoteAudioElement() {
    if (!remoteAudioElement) return
    remoteAudioElement.pause()
    remoteAudioElement.srcObject = null
    remoteAudioElement = null
  }

  function playbackErrorMessage(e: unknown): string {
    if (e instanceof DOMException) return `${e.name}: ${e.message}`
    if (e instanceof Error) return e.message
    return String(e)
  }

  async function playElement(el: HTMLMediaElement, label: string) {
    el.muted = isOutputMuted.value
    el.volume = isOutputMuted.value ? 0 : 1
    try {
      await el.play()
      pushNote(`play ok (${label})`)
      return true
    } catch (e: unknown) {
      pushNote(`play blocked (${label}): ${playbackErrorMessage(e)}`)
      return false
    }
  }

  async function ensurePlayback(reason: string) {
    applyOutputMuted()
    const targets: Array<{ el: HTMLMediaElement; label: string }> = []
    const videoEl = videoElement.value
    const hasDedicatedAudioOutput = dedicatedAudioOutput && !!remoteAudioElement?.srcObject
    if (!hasDedicatedAudioOutput && videoEl?.srcObject) {
      targets.push({ el: videoEl, label: `video ${reason}` })
    }
    if (remoteAudioElement?.srcObject) {
      targets.push({ el: remoteAudioElement, label: `audio ${reason}` })
    }
    if (targets.length === 0) return

    const results = await Promise.all(targets.map((target) => playElement(target.el, target.label)))
    needsPlaybackGesture.value = !isOutputMuted.value && results.some((ok) => !ok)
  }

  async function resumePlayback() {
    isOutputMuted.value = false
    applyOutputMuted()
    await ensurePlayback('user gesture')
  }

  async function toggleOutputMute() {
    isOutputMuted.value = !isOutputMuted.value
    applyOutputMuted()
    pushNote(`assistant output ${isOutputMuted.value ? 'muted' : 'unmuted'}`)
    if (isOutputMuted.value) {
      needsPlaybackGesture.value = false
      return
    }
    needsPlaybackGesture.value = false
    await ensurePlayback('assistant output unmuted')
  }

  async function pollNetworkStats() {
    if (!pc) return
    try {
      const stats = await pc.getStats()
      const net = emptyNetworkStats()

      stats.forEach((report) => {
        if (report.type === 'inbound-rtp' && report.kind === 'video') {
          net.framesDecoded = report.framesDecoded ?? 0
          net.framesDropped = report.framesDropped ?? 0
          net.frameWidth = report.frameWidth ?? 0
          net.frameHeight = report.frameHeight ?? 0
          net.packetsLost = report.packetsLost ?? 0
          net.packetsReceived = report.packetsReceived ?? 0
          net.bytesReceived = report.bytesReceived ?? 0
          net.jitterMs = report.jitter != null ? Math.round(report.jitter * 1000 * 10) / 10 : null
          net.nackCount = report.nackCount ?? 0
          net.pliCount = report.pliCount ?? 0
          net.firCount = report.firCount ?? 0
          net.jitterBufferDelayMs = report.jitterBufferDelay != null && report.jitterBufferEmittedCount
            ? Math.round((report.jitterBufferDelay / report.jitterBufferEmittedCount) * 1000 * 10) / 10
            : null
          net.jitterBufferEmittedCount = report.jitterBufferEmittedCount ?? 0
          if (net.packetsReceived > 0) {
            net.lossRate = Math.round((net.packetsLost / (net.packetsLost + net.packetsReceived)) * 10000) / 10000
          }
          if (report.codecId) {
            const codecReport = stats.get(report.codecId)
            if (codecReport) {
              net.codec = codecReport.mimeType ?? ''
            }
          }
        }
        if (report.type === 'candidate-pair' && report.state === 'succeeded') {
          net.roundTripTimeMs = report.currentRoundTripTime != null
            ? Math.round(report.currentRoundTripTime * 1000 * 10) / 10
            : null
        }
      })
      debugState.value.network = net
      if (!avSyncLoggingEnabled) {
        debugState.value.avSync = null
        return
      }
      const avSync = estimateAVPlayoutFromStats(stats, avPlayoutEstimator)
      debugState.value.avSync = avSync
      if (!avSync.active || avSync.jitterBufferDelayDeltaMs === null) {
        return
      }
      const now = Date.now()
      if (now - lastAVSyncLogAtMs >= 1000) {
        lastAVSyncLogAtMs = now
        console.log(`[DirectRTC][${ts()}] ${formatJitterBufferDelta(avSync)}`)
      }
    } catch {
      // getStats can fail during reconnection, ignore
    }
  }

  /**
   * Connect to the server via Direct P2P WebRTC.
   * Only acquires microphone and sends webrtc_ready.
   * PeerConnection is created later when the server sends webrtc_offer.
   * @param signalingFn - function to send signaling messages via WebSocket
   */
  async function connect(
    signalingFn: (msg: any) => void,
    options: { dedicatedAudioOutput?: boolean; avCalibration?: boolean } = {},
  ) {
    if (connectionState.value === 'connecting' || connectionState.value === 'connected') {
      return
    }

    connectionState.value = 'connecting'
    debugState.value = { ...emptyDebugState(), connectionState: 'connecting' }
    error.value = ''
    resetState()
    sendSignaling = signalingFn
    pendingIceServers = null
    sentWebrtcReady = false
    needsPlaybackGesture.value = false
    isOutputMuted.value = false
    avSyncLoggingEnabled = false
    avSegmentPresentationLoggingEnabled = false
    avPlayoutEstimator = createAVPlayoutEstimatorState()
    lastAVSyncLogAtMs = 0
    resetAVSegmentDiagnostics()
    configureAVCalibration(!!options.avCalibration)
    dedicatedAudioOutput = !!options.dedicatedAudioOutput
    clearRemoteAudioElement()

    // Try to acquire mic, but don't abort the connection if it fails.
    // The WebRTC session (AI video/audio) still works without a mic track.
    // The user can retry via the mic button once OS/browser permission is granted.
    if (!localStream) {
      try {
        await requestMicrophone('connect')
      } catch (micErr: any) {
        pushNote(`mic failed on connect: ${micErr?.name ?? micErr}`)
      }
    } else {
      for (const track of localStream.getAudioTracks()) {
        attachMicMeter(track)
      }
    }

    try {
      // Tell server we're ready for negotiation
      sendSignaling({ type: 'av_calibration_config', enabled: !!options.avCalibration })
      sendSignaling({ type: 'webrtc_ready' })
      sentWebrtcReady = true
      pushNote('sent webrtc_ready')
    } catch (e: unknown) {
      stopMicMetering()
      const msg = e instanceof Error ? e.message : 'Connection failed'
      error.value = msg
      connectionState.value = 'error'
      debugState.value.connectionState = 'error'
      pushNote(`connect error: ${msg}`)
    }
  }

  /**
   * Create PeerConnection with the given ICE servers and set up handlers.
   */
  function createPeerConnection(iceServers: RTCIceServer[]): RTCPeerConnection {
    const newPc = new RTCPeerConnection({ iceServers })

    // Handle remote tracks (video + audio from server)
    let videoMST: MediaStreamTrack | null = null
    let audioMST: MediaStreamTrack | null = null

    const mergeStream = (reason: string) => {
      const el = videoElement.value
      if (!el || (!videoMST && !audioMST)) return
      const tracks: MediaStreamTrack[] = []
      if (videoMST) tracks.push(videoMST)
      if (audioMST && !dedicatedAudioOutput) tracks.push(audioMST)
      el.srcObject = new MediaStream(tracks)
      el.muted = isOutputMuted.value
      el.volume = isOutputMuted.value ? 0 : 1
      console.log(`[DirectRTC][${ts()}] merged stream set: video=${!!videoMST} audio=${!!audioMST}`)
      if (audioMST && dedicatedAudioOutput) {
        const audioEl = remoteAudioEl()
        audioEl.srcObject = new MediaStream([audioMST])
        applyOutputMuted()
      }
      // srcObject may be replaced when audio arrives after video, so autoplay is not enough.
      void ensurePlayback(reason)
    }

    newPc.ontrack = (event) => {
      const track = event.track
      const now = Date.now()

      if (track.kind === 'video') {
        console.log(`[DirectRTC][${ts()}] VIDEO track received`)
        debugState.value.videoSubscribedAtMs = now
        pushNote(`video track received`)
        track.onunmute = () => {
          debugState.value.videoUnmutedAtMs = Date.now()
          pushNote('video track unmuted')
        }
        videoMST = track
        if (videoElement.value) {
          mergeStream('video track')
          attachVideoFrameCallback(videoElement.value)
        }
      }

      if (track.kind === 'audio') {
        console.log(`[DirectRTC][${ts()}] AUDIO track received`)
        debugState.value.audioSubscribedAtMs = now
        pushNote('audio track received')
        track.onunmute = () => {
          debugState.value.audioUnmutedAtMs = Date.now()
          pushNote('audio track unmuted')
        }
        audioMST = track
        remoteAudioTrackForCalibration = track
        if (avCalibrationEnabled) {
          void startAVCalibrationAudioMonitor(track)
        }

        const el = videoElement.value
        if (el) {
          el.addEventListener('play', () => {
            if (audioPlayWallMs === null) {
              audioPlayWallMs = performance.now()
              debugState.value.firstPlayAtMs = Date.now()
              console.log(`[DirectRTC][${ts()}] first play: currentTime=${el.currentTime.toFixed(3)}s`)
            }
          }, { once: true })
        }

        mergeStream('audio track')
      }
    }

    newPc.onicecandidate = (event) => {
      if (event.candidate) {
        sendSignaling?.({
          type: 'ice_candidate',
          candidate: event.candidate.candidate,
          sdp_mid: event.candidate.sdpMid,
          sdp_mline_index: event.candidate.sdpMLineIndex,
        })
      }
    }

    newPc.onconnectionstatechange = () => {
      // Guard against stale callbacks after a mic-retry reconnect.
      if (newPc !== pc) return
      const state = newPc.connectionState
      console.log(`[DirectRTC][${ts()}] connection state: ${state}`)
      pushNote(`connection: ${state}`)
      if (state === 'connected') {
        connectionState.value = 'connected'
        debugState.value.connectionState = 'connected'
      } else if (state === 'failed' || state === 'closed') {
        connectionState.value = 'disconnected'
        debugState.value.connectionState = 'disconnected'
      }
    }

    // Add microphone tracks
    if (localStream) {
      for (const track of localStream.getAudioTracks()) {
        newPc.addTrack(track, localStream)
      }
    }

    // Start network stats polling
    if (networkStatsTimer) clearInterval(networkStatsTimer)
    networkStatsTimer = setInterval(() => void pollNetworkStats(), 1000)

    return newPc
  }

  /**
   * Handle incoming signaling messages from the server (via WebSocket).
   * Operations are serialized so that addIceCandidate always waits for
   * a pending setRemoteDescription to complete first.
   */
  function handleSignaling(data: any) {
    signalingChain = signalingChain.then(async () => {
      if (data.type === 'webrtc_config') {
        // Save TURN/STUN ICE servers from server; PC will be created on webrtc_offer
        pendingIceServers = data.ice_servers || null
        console.log(`[DirectRTC][${ts()}] received webrtc_config, ice_servers:`, pendingIceServers)
        pushNote('received webrtc_config')
        return
      }

      if (data.type === 'av_segment_diagnostic') {
        handleAVSegmentDiagnostic(data)
        return
      }

      if (data.type === 'webrtc_offer') {
        // Create PeerConnection now, using TURN ICE servers if available
        const iceServers: RTCIceServer[] = pendingIceServers || [{ urls: 'stun:stun.l.google.com:19302' }]
        console.log(`[DirectRTC][${ts()}] creating PeerConnection with ICE servers:`, iceServers)
        pc = createPeerConnection(iceServers)

        console.log(`[DirectRTC][${ts()}] received SDP offer, setting remote description...`)
        await pc.setRemoteDescription(new RTCSessionDescription({
          type: 'offer',
          sdp: data.sdp,
        }))
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        sendSignaling?.({
          type: 'webrtc_answer',
          sdp: answer.sdp,
        })
        console.log(`[DirectRTC][${ts()}] SDP answer sent`)
        pushNote('SDP answer sent')
        return
      }

      if (data.type === 'ice_candidate' && data.candidate) {
        if (!pc) return
        console.log(`[DirectRTC][${ts()}] adding remote ICE candidate: ${data.candidate.substring(0, 80)}...`)
        try {
          await pc.addIceCandidate(new RTCIceCandidate({
            candidate: data.candidate,
            sdpMid: data.sdp_mid ?? undefined,
            sdpMLineIndex: data.sdp_mline_index ?? undefined,
          }))
          console.log(`[DirectRTC][${ts()}] ICE candidate added successfully`)
        } catch (e) {
          console.warn('[DirectRTC] addIceCandidate failed:', e)
        }
      }
    }).catch(e => {
      console.error('[DirectRTC] signaling chain error:', e)
    })
  }

  async function toggleMute() {
    // If mic not yet acquired, retry on user gesture.
    if (!localStream) {
      error.value = ''
      try {
        await requestMicrophone('click')
        if (sentWebrtcReady && sendSignaling && connectionState.value === 'connected') {
          // Connection already established without mic — close the current PC and
          // re-negotiate so the new mic track is included. Only safe when fully
          // connected (not mid-negotiation).
          pc?.close()
          pc = null
          sentWebrtcReady = false
          signalingChain = Promise.resolve()
          connectionState.value = 'connecting'
          debugState.value.connectionState = 'connecting'
          sendSignaling({ type: 'webrtc_ready' })
          sentWebrtcReady = true
          pushNote('mic acquired, reconnecting with audio')
        } else if (!sentWebrtcReady && sendSignaling) {
          sendSignaling({ type: 'webrtc_ready' })
          sentWebrtcReady = true
          pushNote('sent webrtc_ready (mic acquired on click)')
        }
      } catch (e: unknown) {
        stopMicMetering()
        const msg = e instanceof Error ? e.message : 'Microphone failed'
        error.value = msg
        pushNote(`mic retry failed: ${msg}`)
      }
      return
    }

    const next = !isMuted.value
    for (const track of localStream.getAudioTracks()) {
      track.enabled = !next
    }
    isMuted.value = next
    pushNote(`mic ${next ? 'muted' : 'unmuted'}`)
  }

  function disconnect() {
    stopMicMetering()
    if (networkStatsTimer) {
      clearInterval(networkStatsTimer)
      networkStatsTimer = null
    }
    sentWebrtcReady = false
    needsPlaybackGesture.value = false
    isOutputMuted.value = false
    avSyncLoggingEnabled = false
    avSegmentPresentationLoggingEnabled = false
    avPlayoutEstimator = createAVPlayoutEstimatorState()
    lastAVSyncLogAtMs = 0
    resetAVSegmentDiagnostics()
    configureAVCalibration(false)
    remoteAudioTrackForCalibration = null

    if (videoElement.value) {
      videoElement.value.srcObject = null
    }
    clearRemoteAudioElement()

    if (localStream) {
      localStream.getTracks().forEach(t => t.stop())
      localStream = null
    }

    resetState()
    pc?.close()
    pc = null
    sendSignaling = null
    signalingChain = Promise.resolve()
    pendingIceServers = null
    dedicatedAudioOutput = false
    connectionState.value = 'disconnected'
    debugState.value.connectionState = 'disconnected'
    if (debugPollTimer) {
      window.clearInterval(debugPollTimer)
      debugPollTimer = null
    }
  }

  // Debug state polling (same pattern as useWebRTC)
  let lastPollFrames = 0
  let lastPollTimeMs = 0

  watch(videoElement, (el) => {
    if (!el) return
    if (debugPollTimer) {
      window.clearInterval(debugPollTimer)
      debugPollTimer = null
    }
    lastPollFrames = 0
    lastPollTimeMs = Date.now()
    debugPollTimer = window.setInterval(() => {
      const quality = typeof el.getVideoPlaybackQuality === 'function'
        ? el.getVideoPlaybackQuality()
        : null
      debugState.value.videoCurrentTime = el.currentTime
      debugState.value.readyState = el.readyState
      debugState.value.playbackRate = el.playbackRate
      const currentFrames = quality?.totalVideoFrames ?? videoFrameCount
      debugState.value.decodedFrames = currentFrames
      debugState.value.droppedFrames = quality?.droppedVideoFrames ?? 0
      debugState.value.totalFrames = currentFrames
      debugState.value.lastVideoFrameAtMs = videoLastFrameWallMs
      const now = Date.now()
      const dt = (now - lastPollTimeMs) / 1000
      if (dt > 0 && lastPollTimeMs > 0) {
        debugState.value.fps = Math.round((currentFrames - lastPollFrames) / dt)
      }
      lastPollFrames = currentFrames
      lastPollTimeMs = now
      if (videoFirstFrameTime !== null && debugState.value.videoFirstFrameAtMs === null) {
        debugState.value.videoFirstFrameAtMs = Date.now()
      }
      debugState.value.jitter = computeJitterStats()
    }, 500)
    el.addEventListener('emptied', () => pushNote('video element emptied'))
    el.addEventListener('waiting', () => pushNote('video element waiting'))
    el.addEventListener('stalled', () => pushNote('video element stalled'))
    el.addEventListener('playing', () => pushNote('video element playing'))
    el.addEventListener('ended', () => {
      if (debugPollTimer) {
        window.clearInterval(debugPollTimer)
        debugPollTimer = null
      }
    }, { once: true })
  })

  return {
    videoElement,
    connectionState,
    debugState,
    error,
    needsPlaybackGesture,
    isOutputMuted,
    isMuted,
    micBarLevels,
    connect,
    disconnect,
    toggleMute,
    resumePlayback,
    toggleOutputMute,
    handleSignaling,
    setAVSyncLoggingEnabled,
  }
}
