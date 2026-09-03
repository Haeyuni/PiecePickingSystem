/** /ws/live 구독 (웹_인터페이스_정의서.md 4절).
 *
 * 재전송 큐가 없으므로(4절), 재연결 시에는 스냅샷 API로 현재 상태를 다시 받는다.
 * 이 훅은 연결 상태와 이벤트 전달만 책임지고, 무엇을 다시 받을지는 호출부가 정한다.
 */
import { useEffect, useRef, useState } from 'react'
import type { LiveEvent } from '../types'

const RECONNECT_DELAY_MS = 2000

export function useLive(onEvent: (event: LiveEvent) => void) {
  const [connected, setConnected] = useState(false)
  // 콜백이 매 렌더 새로 만들어져도 소켓을 다시 열지 않도록 ref로 최신값만 들고 있는다
  const handler = useRef(onEvent)
  handler.current = onEvent

  useEffect(() => {
    let socket: WebSocket | null = null
    let timer: number | undefined
    let closed = false

    const connect = () => {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${scheme}://${window.location.host}/ws/live`)

      socket.onopen = () => setConnected(true)
      socket.onmessage = (message) => {
        try {
          handler.current(JSON.parse(message.data) as LiveEvent)
        } catch {
          // 해석 불가 메시지는 버린다 — 화면을 멈추게 두지 않는다
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!closed) timer = window.setTimeout(connect, RECONNECT_DELAY_MS)
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      closed = true
      if (timer) window.clearTimeout(timer)
      socket?.close()
    }
  }, [])

  return { connected }
}
