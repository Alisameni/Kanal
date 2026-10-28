import type { ChatMessage, ChatTurn } from '../types'

/**
 * How many history messages the frontend is allowed to send.
 *
 * Mirrors `HISTORY_LIMIT` in main.py (and `MAX_HISTORY_MESSAGES` in server.py):
 * anything longer would only be trimmed away by the backend, so the browser
 * never uploads more than the server is willing to keep.
 */
export const HISTORY_LIMIT = 10

/**
 * Turn the on-screen transcript into the `history` array `POST /chat` expects.
 *
 * Two rules matter here:
 *
 * - Messages flagged as `failed` are dropped: the model must never see a turn
 *   that never actually succeeded.
 * - At most the last `HISTORY_LIMIT` messages are kept, so the request stays
 *   bounded.
 *
 * `upTo` defaults to the whole list. A retry passes the index of the failed
 * message so only the valid history *before* it is rebuilt.
 */
export function buildHistory(
  messages: ChatMessage[],
  upTo: number = messages.length,
): ChatTurn[] {
  return messages
    .slice(0, upTo)
    .filter((message) => !message.failed)
    .slice(-HISTORY_LIMIT)
    .map((message) => ({ role: message.role, content: message.content }))
}
