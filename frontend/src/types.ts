/**
 * Shared types for the Kanal chat UI.
 *
 * The `system` role is intentionally absent: the backend owns the system
 * prompt and adds it on every request, so the frontend can only ever speak as
 * `user` or `assistant`.
 */

export type ChatRole = 'user' | 'assistant'

/**
 * One message kept in the UI state. `id` is only used for rendering keys (and to
 * revive a failed turn in place on retry).
 *
 * `failed` marks a user turn whose request did not succeed: the text stays on
 * screen so the user does not lose it, but it is filtered out of `history` and
 * therefore never sent to the model.
 */
export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  failed?: boolean
}

/** One conversation turn in the shape `POST /chat` expects in `history`. */
export interface ChatTurn {
  role: ChatRole
  content: string
}

/** The JSON body returned by a successful `POST /chat`. */
export interface ChatApiResponse {
  reply: string
  history: ChatTurn[]
}
