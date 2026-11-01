import { useCallback, useRef, useState } from 'react'

import { ApiError, cancelResearch, saveResearch, sendChat } from '../api/chat'
import { buildHistory } from '../lib/history'
import type { ChatMessage, ChatTurn, PendingSave, ResearchSource } from '../types'

/** Request lifecycle of the conversation panel. */
export type ChatStatus = 'idle' | 'pending' | 'error'

/** Whether the backend has been reached successfully so far. */
export type ConnectionState = 'unknown' | 'online' | 'offline'

/**
 * Lifecycle of the save proposal attached to the newest reply.
 *
 * It mirrors the backend's own pending store: a proposal waits (`pending`), is
 * handed to the writer of exactly one request (`saving`), and ends as `saved`,
 * `cancelled`, or -- when its id is gone, expired or already used -- as `error`.
 */
export type SaveProposalStatus =
  | 'pending'
  | 'saving'
  | 'saved'
  | 'cancelling'
  | 'cancelled'
  | 'error'

/**
 * Everything the confirmation card renders, plus the state of its own request.
 *
 * The proposal text is copied out of the reply once and never mutated, so the
 * card always shows exactly what the backend offered. The `saveId` is the only
 * thing the UI may send back.
 */
export interface SaveProposal {
  saveId: string
  title: string
  summary: string
  filename: string
  sources: ResearchSource[]
  /** The Markdown that would be written; null when the backend sent no preview. */
  preview: string | null
  status: SaveProposalStatus
  /** Persian failure text; null until something actually failed. */
  error: string | null
  /**
   * True when a failure may be retried because the proposal is still waiting
   * with the backend. A 404 leaves the id unusable, so it stays false.
   */
  retryable: boolean
}

export interface UseChatResult {
  messages: ChatMessage[]
  status: ChatStatus
  connection: ConnectionState
  error: string | null
  /** The save proposal of the newest reply, or null when there is none. */
  proposal: SaveProposal | null
  sendMessage: (text: string) => void
  retryMessage: (id: string) => void
  clearChat: () => void
  dismissError: () => void
  confirmSave: () => void
  cancelSave: () => void
  dismissProposal: () => void
}

function newId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`
}
/** Words for a failure that did not come from the API layer. */
const UNKNOWN_ERROR = 'خطای نامشخصی رخ داد.'

/**
 * True while a proposal can still be confirmed or cancelled.
 *
 * `saving` and `cancelling` count as open on purpose: the card stays where it is,
 * with its buttons disabled, until its request settles, so the outcome of a
 * click can never be lost by a rerender.
 */
export function isProposalOpen(proposal: SaveProposal): boolean {
  return (
    proposal.status === 'pending' ||
    proposal.status === 'saving' ||
    proposal.status === 'cancelling' ||
    (proposal.status === 'error' && proposal.retryable)
  )
}

/** True while a decision request of this proposal is on its way. */
export function isProposalBusy(proposal: SaveProposal): boolean {
  return proposal.status === 'saving' || proposal.status === 'cancelling'
}

/**
 * Whether a failed decision request is worth repeating.
 *
 * The backend hands a proposal back to its store when the write itself fails
 * (500) and also when the request never reached it at all (no status), so both
 * are retryable. A 404 is final instead: the proposal expired, was already saved
 * or was already cancelled, and that id can never become usable again.
 */
function isRetryableDecisionFailure(status: number | null): boolean {
  return status === null || status === 500
}

/** Copy one reply's proposal into the state the card renders. */
function toProposal(pending: PendingSave): SaveProposal {
  return {
    saveId: pending.save_id,
    title: pending.title,
    summary: pending.summary,
    filename: pending.filename,
    sources: pending.sources,
    preview: pending.preview ?? null,
    status: 'pending',
    error: null,
    retryable: false,
  }
}

/**
 * The card to show after a reply arrived.
 *
 * A new proposal always replaces the previous one, because the newest turn is
 * what the backend just prepared. A reply *without* a proposal only clears a
 * card that is already decided: an open one stays on screen, so a stray message
 * can never lose a proposal that is still waiting for a decision.
 */
function nextProposal(
  current: SaveProposal | null,
  pending: PendingSave | undefined,
): SaveProposal | null {
  if (pending) {
    return toProposal(pending)
  }
  return current && isProposalOpen(current) ? current : null
}

/**
 * Owns the whole conversation: the transcript shown on screen *is* the source of
 * the history sent with the next message. There is no server-side session and no
 * stored history, so clearing the state is the only "reset" the UI needs.
 *
 * A failed turn stays in the transcript (so the user does not lose their text)
 * but is flagged `failed`, which keeps it out of every future `history` (see
 * buildHistory). Only the newest turn may be retried, in place, with its own id.
 *
 * The hook also owns the save proposal of the newest reply, because deciding on
 * it is part of the same conversation: the card's state, the two decision calls
 * and their guards all live here, so a component only ever renders and calls.
 */
export function useChat(): UseChatResult {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [status, setStatus] = useState<ChatStatus>('idle')
  const [connection, setConnection] = useState<ConnectionState>('unknown')
  const [error, setError] = useState<string | null>(null)
  const [proposal, setProposal] = useState<SaveProposal | null>(null)
  const inFlight = useRef(false)
  // Guards the two decision calls: a second click -- or a click on the other
  // button while the first request is still out -- must never reach the backend.
  const deciding = useRef(false)
  // Bumped by clearChat(). A reply whose generation no longer matches belongs to
  // a conversation the user has discarded, so it is dropped, never appended.
  const generation = useRef(0)

  /**
   * Apply a change to the proposal that is on screen, and only to that one.
   *
   * Matching on `saveId` means a late answer for a proposal the user has already
   * replaced can never overwrite the card of a newer one, and a card that is
   * already gone can never be resurrected.
   */
  const patchProposal = useCallback(
    (saveId: string, changes: Partial<SaveProposal>) => {
      setProposal((current) =>
        current && current.saveId === saveId
          ? { ...current, ...changes }
          : current,
      )
    },
    [],
  )

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
          // A turn may end with a save proposal; the card follows the reply.
          setProposal((current) => nextProposal(current, response.pending_save))
          setConnection('online')
          setStatus('idle')
        })
        .catch((cause: unknown) => {
          if (token !== generation.current) return
          const failure = cause instanceof ApiError ? cause : null
          setError(failure?.message ?? UNKNOWN_ERROR)
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

  /**
   * Confirm the proposal on screen, which is the only way a file is ever written.
   *
   * Nothing happens without this call: the card is a proposal and the backend
   * writes only after a successful `POST /save`. Both controls are disabled by
   * moving the proposal to `saving` before the request leaves, and the ref guard
   * makes a double click harmless even before that rerender.
   */
  const confirmSave = useCallback(() => {
    if (!proposal || deciding.current || !isProposalOpen(proposal)) {
      return
    }

    const { saveId, filename } = proposal
    deciding.current = true
    patchProposal(saveId, { status: 'saving', error: null, retryable: false })

    saveResearch(saveId)
      .then((result) => {
        patchProposal(saveId, {
          status: 'saved',
          // The backend answers with the name it actually wrote.
          filename: result.filename || filename,
          error: null,
          retryable: false,
        })
      })
      .catch((cause: unknown) => {
        const failure = cause instanceof ApiError ? cause : null
        patchProposal(saveId, {
          status: 'error',
          error: failure?.message ?? UNKNOWN_ERROR,
          // A 500 leaves the proposal in the backend's store (while its TTL is
          // running), so trying again is worthwhile; a 404 does not.
          retryable: failure ? isRetryableDecisionFailure(failure.status) : false,
        })
      })
      .finally(() => {
        deciding.current = false
      })
  }, [patchProposal, proposal])

  /**
   * Drop the proposal on screen, so it can never be written.
   *
   * A proposal that is still open may be cancelled as well: that is the only way
   * to release it on the backend before its TTL runs out.
   */
  const cancelSave = useCallback(() => {
    if (!proposal || deciding.current || !isProposalOpen(proposal)) {
      return
    }

    const { saveId } = proposal
    deciding.current = true
    patchProposal(saveId, { status: 'cancelling', error: null, retryable: false })

    cancelResearch(saveId)
      .then(() => {
        patchProposal(saveId, {
          status: 'cancelled',
          error: null,
          retryable: false,
        })
      })
      .catch((cause: unknown) => {
        const failure = cause instanceof ApiError ? cause : null
        patchProposal(saveId, {
          status: 'error',
          error: failure?.message ?? UNKNOWN_ERROR,
          retryable: failure ? isRetryableDecisionFailure(failure.status) : false,
        })
      })
      .finally(() => {
        deciding.current = false
      })
  }, [patchProposal, proposal])

  /** Close a proposal that is already decided; an open one cannot be dismissed. */
  const dismissProposal = useCallback(() => {
    setProposal((current) =>
      current && isProposalOpen(current) ? current : null,
    )
  }, [])

  const clearChat = useCallback(() => {
    // Invalidate any in-flight reply so it cannot land in the (now empty)
    // conversation, and free the panel straight away so a new conversation can
    // start without waiting for the discarded request to settle.
    generation.current += 1
    inFlight.current = false
    deciding.current = false
    setMessages([])
    setError(null)
    setStatus('idle')
    // The proposal belongs to the conversation that is being discarded. Its id
    // simply expires in the backend, so nothing else has to be told.
    setProposal(null)
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
    proposal,
    sendMessage,
    retryMessage,
    clearChat,
    dismissError,
    confirmSave,
    cancelSave,
    dismissProposal,
  }
}
