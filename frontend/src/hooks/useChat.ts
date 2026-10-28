import { useCallback, useRef, useState } from 'react'

import { ChatApiError, sendChat } from '../api/chat'
import { buildHistory } from '../lib/history'
import type { ChatMessage, ChatTurn } from '../types'

/** Request lifecycle of the conversation panel. */
export type ChatStatus = 'idle' | 'pending' | 'error'

/** Whether the backend has been reached successfully so far. */
export type ConnectionState = 'unknown' | 'online' | 'offline'

export interface UseChatResult {
  messages: ChatMessage[]
  status: ChatStatus
  connection: ConnectionState
  error: string | null
  sendMessage: (text: string) => void
  retryMessage: (id: string) => void
  clearChat: () => void
  dismissError: () => void
}

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

/**
 * Owns the whole conversation: the transcript shown on screen *is* the source of
 * the history sent with the next message. There is no server-side session and no
 * stored history, so clearing the state is the only "reset" the UI needs.
 *
 * A failed turn stays in the transcript (so the user does not lose their text)
 * but is flagged `failed`, which keeps it out of every future `history` (see
 * buildHistory). Only the newest turn may be retried, in place, with its own id.
 */
export function useChat(): UseChatResult {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [status, setStatus] = useState<ChatStatus>('idle')
  const [connection, setConnection] = useState<ConnectionState>('unknown')
  const [error, setError] = useState<string | null>(null)
  const inFlight = useRef(false)
  // Bumped by clearChat(). A reply whose generation no longer matches belongs to
  // a conversation the user has discarded, so it is dropped, never appended.
  const generation = useRef(0)

  // The single path every request takes, so "send" and "retry" cannot drift.
  const request = useCallback(
    (messageId: string, content: string, history: ChatTurn[]) => {
      const token = generation.current
      inFlight.current = true
      setError(null)
      setStatus('pending')

      sendChat(content, history)
        .then((response) => {
          if (token !== generation.current) return
          setMessages((current) => [
            ...current,
            { id: newId(), role: 'assistant', content: response.reply },
          ])
          setConnection('online')
          setStatus('idle')
        })
        .catch((cause: unknown) => {
          if (token !== generation.current) return
          const failure = cause instanceof ChatApiError ? cause : null
          setError(failure?.message ?? 'خطای نامشخصی رخ داد.')
          // A null status means the request never reached the server.
          setConnection(failure && failure.status === null ? 'offline' : 'online')
          setStatus('error')
          // Keep the user's text, but flag the turn so buildHistory drops it.
          setMessages((current) =>
            current.map((message) =>
              message.id === messageId ? { ...message, failed: true } : message,
            ),
          )
        })
        .finally(() => {
          // Only the live generation owns the flag: a discarded request must not
          // release it on behalf of a newer request that clearChat already let in.
          if (token === generation.current) {
            inFlight.current = false
          }
        })
    },
    [],
  )

  const sendMessage = useCallback(
    (text: string) => {
      const content = text.trim()
      if (!content || inFlight.current) {
        return
      }

      const history = buildHistory(messages)
      const id = newId()
      setMessages((current) => [...current, { id, role: 'user', content }])
      request(id, content, history)
    },
    [messages, request],
  )

  const retryMessage = useCallback(
    (id: string) => {
      if (inFlight.current) {
        return
      }

      const index = messages.findIndex((message) => message.id === id)
      const target = messages[index]
      // Only a flagged message may be revived, and only when it is the newest
      // turn: re-sending an older failed turn would splice a fresh reply into the
      // middle of the conversation, so it is deliberately left as-is.
      if (!target || !target.failed || index !== messages.length - 1) {
        return
      }

      // Only the valid history *before* the failed turn is rebuilt.
      const history = buildHistory(messages, index)
      // Revive in place: same id, same bubble, no duplicate.
      setMessages((current) =>
        current.map((message) =>
          message.id === id ? { ...message, failed: false } : message,
        ),
      )
      request(id, target.content, history)
    },
    [messages, request],
  )

  const clearChat = useCallback(() => {
    // Invalidate any in-flight reply so it cannot land in the (now empty)
    // conversation, and free the panel straight away so a new conversation can
    // start without waiting for the discarded request to settle.
    generation.current += 1
    inFlight.current = false
    setMessages([])
    setError(null)
    setStatus('idle')
  }, [])

  const dismissError = useCallback(() => {
    setError(null)
    setStatus((current) => (current === 'error' ? 'idle' : current))
  }, [])

  return {
    messages,
    status,
    connection,
    error,
    sendMessage,
    retryMessage,
    clearChat,
    dismissError,
  }
}
