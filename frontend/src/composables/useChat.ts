import { ref, computed } from 'vue'
import { getConversationMessages, getTaskEvents, listSessionTasks, sendMessage } from '../services/api'
import { translate } from '../i18n'

export interface ChatMessage {
  id?: string  // Optional ID for deduplication
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  isHistory?: boolean
  sessionId?: string
  artifactUrl?: string
}

export type AvatarStatus = 'idle' | 'speaking' | 'processing'

export function useChat(sessionId: () => string) {
  const ws = ref<WebSocket | null>(null)
  const messages = ref<ChatMessage[]>([])
  const currentTranscript = ref('')

  // New state variables for separate pipeline tracking
  const currentVoiceResponse = ref('')      // For transcript events (voice pipeline)
  const currentTextResponse = ref('')        // For llm_token events (text pipeline)
  const activeResponseId = ref<string>('')   // Track active response to prevent duplicates
  const pipelineMode = ref<'text' | 'voice' | null>(null)  // Track active pipeline

  const latestTurnSeq = ref(0)
  const voiceDrafts = new Map<string, string>()
  const lastTaskSeqByTaskId = new Map<string, number>()

  // Computed property to show the appropriate response based on active pipeline
  const currentLLMResponse = computed(() => {
    // Show whichever has content, prioritizing the active pipeline
    if (pipelineMode.value === 'voice') {
      return currentVoiceResponse.value || currentTextResponse.value
    }
    return currentTextResponse.value || currentVoiceResponse.value
  })

  const avatarStatus = ref<AvatarStatus>('idle')
  const idleVideoUrls = ref<string[]>([])
  const idleVideoUrl = computed(() => idleVideoUrls.value.length > 0 ? idleVideoUrls.value[0] : '')
  const isConnected = ref(false)

  // Signaling handler for Direct WebRTC mode
  let signalingHandler: ((data: any) => void) | null = null

  function resetTransientState() {
    pipelineMode.value = null
    activeResponseId.value = ''
    currentVoiceResponse.value = ''
    currentTextResponse.value = ''
    currentTranscript.value = ''
  }

  function parseTurnSeq(data: any): number {
    const turnSeq = Number(data?.turn_seq ?? 0)
    return Number.isFinite(turnSeq) && turnSeq > 0 ? turnSeq : 0
  }

  function beginTurnSeq(turnSeq: number) {
    if (!turnSeq) return
    if (turnSeq > latestTurnSeq.value) {
      latestTurnSeq.value = turnSeq
      resetTransientState()
    }
  }

  function isOlderTurn(turnSeq: number): boolean {
    return turnSeq > 0 && turnSeq < latestTurnSeq.value
  }

  function upsertMessage(message: ChatMessage, afterId?: string) {
    if (message.id) {
      const existingIndex = messages.value.findIndex(m => m.id === message.id)
      if (existingIndex >= 0) {
        messages.value[existingIndex] = { ...messages.value[existingIndex], ...message }
        return
      }
    }

    if (afterId) {
      const anchorIndex = messages.value.findIndex(m => m.id === afterId)
      if (anchorIndex >= 0) {
        messages.value.splice(anchorIndex + 1, 0, message)
        return
      }
    }

    messages.value.push(message)
  }

  function clearActiveResponse(responseId: string) {
    if (activeResponseId.value === responseId) {
      resetTransientState()
    }
  }

  function registerSignalingHandler(fn: (data: any) => void) {
    signalingHandler = fn
  }

  function handleTaskEvent(data: any, fallbackTask?: any) {
    const task = data.task || fallbackTask || {}
    const taskId = data.task_id || task.id || ''
    const seq = Number(data.seq ?? 0)
    if (taskId && Number.isFinite(seq) && seq > 0) {
      const previousSeq = lastTaskSeqByTaskId.get(taskId) || 0
      if (seq <= previousSeq) return
      lastTaskSeqByTaskId.set(taskId, seq)
    }
    const title = task.title || '后台任务'
    const status = data.status || task.status || ''
    const progress = Number(data.progress ?? task.progress ?? 0)
    const message = data.message || '任务状态已更新。'
    const payload = data.payload || {}
    const artifactId = payload.artifact_id
    const artifactUrl = taskId && artifactId ? `/api/v1/tasks/${taskId}/artifacts/${artifactId}` : undefined
    const suffix = artifactUrl ? '\n资料已生成。' : ''
    upsertMessage({
      id: taskId ? `task-${taskId}` : `task-${Date.now()}`,
      role: 'system',
      content: `任务「${title}」${status}，进度 ${progress}%：${message}${suffix}`,
      timestamp: Date.now(),
      artifactUrl,
    })
  }

  async function recoverTaskEvents() {
    const sid = sessionId()
    if (!sid) return
    try {
      const resp = await listSessionTasks(sid)
      await Promise.all((resp.tasks || []).map(async (task) => {
        const afterSeq = lastTaskSeqByTaskId.get(task.id) || 0
        const eventsResp = await getTaskEvents(task.id, afterSeq)
        for (const event of eventsResp.events || []) {
          handleTaskEvent(event, task)
        }
      }))
    } catch (err) {
      console.warn('[useChat] Failed to recover task events:', err)
    }
  }

  function sendSignaling(msg: any) {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
    ws.value.send(JSON.stringify(msg))
  }

  function sendWSMessage(msg: any): boolean {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return false
    ws.value.send(JSON.stringify(msg))
    return true
  }

  function connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsBase = import.meta.env.VITE_WS_BASE || `ws://${window.location.host}`
      const url = `${wsBase}/ws/chat/${sessionId()}`
      ws.value = new WebSocket(url)

      ws.value.onopen = () => {
        isConnected.value = true
        void recoverTaskEvents()
        resolve()
      }

      ws.value.onclose = () => {
        isConnected.value = false
      }

      ws.value.onerror = (e) => {
        reject(e)
      }

    ws.value.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'transcript': {
          const turnSeq = parseTurnSeq(data)
          const olderTurn = isOlderTurn(turnSeq)
          if (!olderTurn) beginTurnSeq(turnSeq)
          const role: ChatMessage['role'] = data.speaker === 'assistant' ? 'assistant' : 'user'

          if (role === 'assistant') {
            const responseId = turnSeq ? `voice-${turnSeq}` : (activeResponseId.value || `voice-${Date.now()}`)

            if (data.is_final) {
              const finalText = data.text || voiceDrafts.get(responseId) || (activeResponseId.value === responseId ? currentVoiceResponse.value : '')
              if (finalText) {
                upsertMessage({
                  id: responseId,
                  role,
                  content: finalText,
                  timestamp: Date.now(),
                }, turnSeq ? `user-${turnSeq}` : undefined)
              }

              voiceDrafts.delete(responseId)
              clearActiveResponse(responseId)
            } else {
              if (olderTurn) break
              const nextText = (voiceDrafts.get(responseId) || '') + (data.text || '')
              voiceDrafts.set(responseId, nextText)
              activeResponseId.value = responseId
              pipelineMode.value = 'voice'
              currentVoiceResponse.value = nextText
            }
          } else {
            if (!olderTurn) {
              currentTranscript.value = data.text
            }
            if (data.is_final) {
              const messageId = turnSeq ? `user-${turnSeq}` : undefined
              upsertMessage({
                id: messageId,
                role,
                content: data.text,
                timestamp: Date.now(),
              })
              if (!olderTurn) {
                currentTranscript.value = ''
              }
            }
          }
          break
        }

        case 'llm_token': {
          const turnSeq = parseTurnSeq(data)
          const olderTurn = isOlderTurn(turnSeq)
          if (!olderTurn) beginTurnSeq(turnSeq)
          const responseId = turnSeq ? `text-${turnSeq}` : (activeResponseId.value || `text-${Date.now()}`)

          if (olderTurn) {
            if (data.is_final && data.accumulated) {
              upsertMessage({
                id: responseId,
                role: 'assistant',
                content: data.accumulated,
                timestamp: Date.now(),
              }, turnSeq ? `user-${turnSeq}` : undefined)
            }
            break
          }

          if (!pipelineMode.value) {
            pipelineMode.value = 'text'
            activeResponseId.value = responseId
          }
          if (pipelineMode.value !== 'text') {
            resetTransientState()
            pipelineMode.value = 'text'
          }
          activeResponseId.value = responseId

          currentTextResponse.value = data.accumulated

          if (data.is_final) {
            if (responseId) {
              upsertMessage({
                id: responseId,
                role: 'assistant',
                content: data.accumulated,
                timestamp: Date.now(),
              }, turnSeq ? `user-${turnSeq}` : undefined)
            }

            currentTextResponse.value = ''
            clearActiveResponse(responseId)
          }
          break
        }

        case 'assistant_message': {
          const turnSeq = parseTurnSeq(data)
          if (!isOlderTurn(turnSeq)) beginTurnSeq(turnSeq)
          const responseId = turnSeq ? `assistant-message-${turnSeq}` : `assistant-message-${Date.now()}`
          upsertMessage({
            id: responseId,
            role: 'assistant',
            content: data.text || data.message || '',
            timestamp: Date.now(),
          })
          break
        }

        case 'task_event': {
          handleTaskEvent(data)
          break
        }

        case 'idle_video_ready':
          if (data.urls && data.urls.length > 0) {
            idleVideoUrls.value = data.urls
          } else if (data.url) {
            idleVideoUrls.value = [data.url]
          }
          break

        case 'avatar_warning':
          console.warn('[CyberVerse]', data.message || data)
          break

        case 'visual_input_error':
        case 'visual_input_unsupported':
          console.warn('[CyberVerse]', data.message || data)
          break

        case 'avatar_status': {
          const turnSeq = parseTurnSeq(data)
          if (isOlderTurn(turnSeq)) {
            break
          }
          beginTurnSeq(turnSeq)
          avatarStatus.value = data.status
          if (data.status === 'idle') {
            resetTransientState()
          }
          break
        }

        case 'webrtc_config':
        case 'webrtc_offer':
        case 'ice_candidate':
          if (signalingHandler) {
            signalingHandler(data)
          }
          break

        default:
          console.warn('Unknown message type:', data.type)
      }
    }
    })
  }

  function sendText(text: string) {
    const trimmed = text.trim()
    if (!trimmed) return

    const sid = sessionId()
    const payload = JSON.stringify({ type: 'text_input', text: trimmed })
    let sentViaWS = false

    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      try {
        ws.value.send(payload)
        sentViaWS = true
      } catch (err) {
        console.error('[useChat] WS send failed, fallback to HTTP:', err)
      }
    }

    if (!sentViaWS) {
      if (!sid) {
        console.error('[useChat] Cannot send text: missing session id')
        messages.value.push({
          role: 'system',
          content: translate('chat.sendFailedNoSession'),
          timestamp: Date.now(),
        })
        return
      }
      void sendMessage(sid, trimmed).catch((err) => {
        console.error('[useChat] HTTP fallback send failed:', err)
        messages.value.push({
          role: 'system',
          content: translate('chat.sendFailedNetwork'),
          timestamp: Date.now(),
        })
      })
    }

    messages.value.push({
      role: 'user',
      content: trimmed,
      timestamp: Date.now(),
    })
  }

  function interrupt() {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
    ws.value.send(JSON.stringify({ type: 'interrupt' }))
  }

  // ── History loading ──
  const historyLoading = ref(false)
  const historyHasMore = ref(false)
  const historyNextCursor = ref('')

  async function loadHistory(characterId: string) {
    if (!characterId || historyLoading.value) return
    historyLoading.value = true
    try {
      const resp = await getConversationMessages(
        characterId,
        50,
        historyNextCursor.value || undefined,
      )
      const historyMessages: ChatMessage[] = resp.messages.map((m) => ({
        role: m.role as ChatMessage['role'],
        content: m.content,
        timestamp: new Date(m.timestamp).getTime() || 0,
        isHistory: true,
        sessionId: m.session_id,
      }))
      // Prepend history before current messages
      messages.value = [...historyMessages, ...messages.value]
      historyNextCursor.value = resp.next_cursor
      historyHasMore.value = resp.has_more
    } catch (e) {
      console.error('[useChat] Failed to load history:', e)
    } finally {
      historyLoading.value = false
    }
  }

  function disconnect() {
    ws.value?.close()
    ws.value = null
    isConnected.value = false
    latestTurnSeq.value = 0
    voiceDrafts.clear()
    resetTransientState()
  }

  return {
    messages,
    currentTranscript,
    currentLLMResponse,  // Now a computed property
    avatarStatus,
    idleVideoUrl,
    idleVideoUrls,
    isConnected,
    historyLoading,
    historyHasMore,
    connect,
    sendText,
    interrupt,
    disconnect,
    loadHistory,
    registerSignalingHandler,
    sendSignaling,
    sendWSMessage,
  }
}
