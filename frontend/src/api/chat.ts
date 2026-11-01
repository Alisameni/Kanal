import { isSafeHttpUrl } from '../lib/url'
import type {
  CancelConfirmation,
  ChatApiResponse,
  ChatTurn,
  PendingSave,
  ResearchSource,
  SaveConfirmation,
} from '../types'

/**
 * The single place the UI talks to the backend.
 *
 * Three endpoints are used, all with a relative path so the Vite dev/preview
 * proxy (see vite.config.ts) can forward them to FastAPI without any CORS setup:
 * `POST /chat` for one turn, `POST /save` to write the research document of a
 * proposal the user confirmed, and `POST /save/cancel` to drop it. No API key,
 * base URL or `.env` value ever reaches the browser.
 *
 * Every call goes through `postJson`, so the transport, the timeout and the
 * error mapping exist exactly once. What differs per endpoint is only the body,
 * the shape of a valid answer (the `validate` argument) and the wording of a few
 * status codes.
 */

const CHAT_ENDPOINT = '/chat'
const SAVE_ENDPOINT = '/save'
const SAVE_CANCEL_ENDPOINT = '/save/cancel'

/**
 * Sits above the backend's worst case so the browser never aborts a request the
 * server is still legitimately working on.
 *
 * A search turn is capped at 100s in main.py (TOOL_DEADLINE_SECONDS): a slow
 * provider can spend up to ~60s on its rounds and searches and then up to 40s
 * writing the answer (TOOL_ANSWER_TIMEOUT_SECONDS). The plain path, with no
 * search key, has no such budget and keeps the client's own 20s timeout with one
 * retry, so it stays far below this.
 *
 * That 100s is the ceiling, not the expected wait: a measured search turn
 * answering a long analytical question finishes in ~35-40s. Aborting here only
 * stops the browser from waiting; it does not guarantee the backend stops working
 * on the request.
 */
const REQUEST_TIMEOUT_MS = 130_000

/**
 * The two decision calls are much cheaper than a turn: they only name a waiting
 * proposal, and the backend answers from its own in-memory store plus one small
 * file write, with no provider call involved. A short ceiling therefore fails
 * fast instead of leaving the card stuck in "saving" for minutes.
 */
const DECISION_TIMEOUT_MS = 15_000

/** Mirrors `MAX_MESSAGE_LENGTH` in server.py: the composer refuses to exceed it. */
export const MAX_MESSAGE_LENGTH = 4000

/** One Persian sentence per failure kind. Nothing raw is ever shown to the user. */
const TIMEOUT_MESSAGE = 'زمان انتظار برای پاسخ سرور به پایان رسید.'
const NETWORK_MESSAGE = 'ارتباط با سرور برقرار نشد. از اجرا بودن Backend مطمئن شوید.'
const INVALID_PAYLOAD_MESSAGE = 'پاسخ سرور قابلخواندن نبود.'

/** Persian, human-readable text for the statuses the backend can return. */
const STATUS_MESSAGES: Record<number, string> = {
  400: 'درخواست معتبر نبود.',
  401: 'دسترسی به سرور مجاز نیست.',
  403: 'دسترسی به این بخش مجاز نیست.',
  404: 'نشانی درخواستشده پیدا نشد.',
  408: 'زمان انتظار برای پاسخ سرور به پایان رسید.',
  422: 'درخواست ارسالشده معتبر نبود.',
  429: 'تعداد درخواستها زیاد است؛ کمی بعد دوباره تلاش کنید.',
  500: 'خطای غیرمنتظرهای در سرور رخ داد.',
  502: 'سرویسدهندهٔ مدل پاسخ معتبری برنگرداند.',
  503: 'سرور در دسترس نیست یا تنظیمات آن کامل نیست.',
  504: 'پاسخ سرور بیش از حد طول کشید؛ دوباره تلاش کنید.',
}

/**
 * Both decision endpoints answer 404 for *every* unusable id (unknown, expired,
 * already saved, already cancelled), so the text says what that means for a
 * proposal instead of falling back to the generic route wording above.
 */
const SAVE_STATUS_MESSAGES: Record<number, string> = {
  404: 'این پیشنهاد دیگر وجود ندارد؛ منقضی شده یا پیشتر استفاده شده است.',
  422: 'درخواست ذخیره معتبر نبود.',
  500: 'ذخیره‌سازی تحقیق ناموفق بود؛ می‌توانید دوباره تلاش کنید.',
}

const CANCEL_STATUS_MESSAGES: Record<number, string> = {
  404: 'این پیشنهاد دیگر وجود ندارد؛ منقضی شده یا پیشتر لغو شده است.',
  422: 'درخواست لغو معتبر نبود.',
  500: 'لغو پیشنهاد ناموفق بود.',
}

/** An error the UI can show verbatim. `status === null` means "never reached the server". */
export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** True when our own timeout, not the network, aborted the request. */
function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

/**
 * Map an HTTP failure to a short Persian sentence. The server's own `detail`
 * body is deliberately ignored so internal text, tracebacks or field names can
 * never reach the UI.
 */
function describeFailure(status: number, overrides?: Record<number, string>): string {
  return (
    overrides?.[status] ??
    STATUS_MESSAGES[status] ??
    `درخواست با خطای ${status} ناموفق بود.`
  )
}

/** Per-endpoint transport settings; everything else about a request is shared. */
interface RequestOptions {
  /** Bounds how long the browser waits; see REQUEST_TIMEOUT_MS. */
  timeoutMs?: number
  /** Status texts that differ from the generic STATUS_MESSAGES table. */
  statusMessages?: Record<number, string>
}

/** Returns the parsed answer, or null when the payload does not match the contract. */
type Validator<T> = (data: unknown) => T | null

/**
 * The one request path of this module.
 *
 * `validate` is what keeps an unexpected payload out of the UI: it turns the
 * parsed JSON into the exact shape the caller may rely on, and returns null when
 * anything essential is missing or of the wrong type.
 */
async function postJson<T>(
  path: string,
  body: unknown,
  validate: Validator<T>,
  options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? REQUEST_TIMEOUT_MS,
  )

  // The timer is cleared once, in the outermost `finally`, so it also protects
  // the body read: `fetch()` resolving only means the response *headers* have
  // arrived, not that the whole body has been read and parsed.
  try {
    let response: Response
    try {
      response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
    } catch (cause) {
      throw new ApiError(
        isAbortError(cause) ? TIMEOUT_MESSAGE : NETWORK_MESSAGE,
      )
    }

    if (!response.ok) {
      throw new ApiError(
        describeFailure(response.status, options.statusMessages),
        response.status,
      )
    }

    let data: unknown
    try {
      data = await response.json()
    } catch (cause) {
      throw new ApiError(
        isAbortError(cause) ? TIMEOUT_MESSAGE : INVALID_PAYLOAD_MESSAGE,
        response.status,
      )
    }

    const result = validate(data)
    if (result === null) {
      throw new ApiError(INVALID_PAYLOAD_MESSAGE, response.status)
    }

    return result
  } finally {
    clearTimeout(timer)
  }
}

/**
 * Payload normalisers.
 *
 * The backend guarantees these shapes, but this is the boundary where a response
 * becomes UI data, so each field is checked once here instead of being trusted
 * later by a component. Nothing is ever interpreted as HTML: every field stays a
 * plain string.
 */

/** A plain object, as opposed to null, an array or a primitive. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The text of one backend field; anything that is not a string reads as "absent". */
function asText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/**
 * Keep only the sources that carry a link the UI is allowed to open.
 *
 * The backend already validates its sources (save_tool.select_sources), but a
 * link that is not absolute http/https is dropped here as well instead of
 * ending up in an anchor. A missing title falls back to the URL, exactly like
 * the backend does.
 */
function asSources(value: unknown): ResearchSource[] {
  if (!Array.isArray(value)) {
    return []
  }

  const sources: ResearchSource[] = []
  for (const item of value) {
    if (!isRecord(item)) {
      continue
    }
    const url = asText(item.url)
    if (!isSafeHttpUrl(url)) {
      continue
    }
    sources.push({ title: asText(item.title) || url, url })
  }
  return sources
}

/**
 * Normalise `pending_save`, or return null when there is nothing to review.
 *
 * An id is what makes a proposal actionable at all, so a payload without one is
 * treated as "no proposal": a card whose only buttons could do nothing but
 * return 404 would be worse than no card.
 */
function asPendingSave(value: unknown): PendingSave | null {
  if (!isRecord(value)) {
    return null
  }

  const saveId = asText(value.save_id)
  if (!saveId) {
    return null
  }

  const proposal: PendingSave = {
    save_id: saveId,
    title: asText(value.title),
    summary: asText(value.summary),
    filename: asText(value.filename),
    sources: asSources(value.sources),
  }

  // `preview` stays absent when the backend did not send one, so the card can
  // simply not offer the toggle.
  const preview = asText(value.preview)
  if (preview) {
    proposal.preview = preview
  }
  return proposal
}

/** The history of a reply; the UI keeps its own transcript, so this is only a contract check. */
function asHistory(value: unknown): ChatTurn[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(
    (turn): turn is ChatTurn =>
      isRecord(turn) &&
      (turn.role === 'user' || turn.role === 'assistant') &&
      typeof turn.content === 'string',
  )
}

function asChatResponse(value: unknown): ChatApiResponse | null {
  if (!isRecord(value) || typeof value.reply !== 'string') {
    return null
  }

  const response: ChatApiResponse = {
    reply: value.reply,
    history: asHistory(value.history),
  }

  const pending = asPendingSave(value.pending_save)
  if (pending) {
    response.pending_save = pending
  }
  return response
}

function asSaveConfirmation(value: unknown): SaveConfirmation | null {
  if (!isRecord(value) || value.status !== 'saved') {
    return null
  }
  return { status: 'saved', filename: asText(value.filename) }
}

function asCancelConfirmation(value: unknown): CancelConfirmation | null {
  if (!isRecord(value) || value.status !== 'cancelled') {
    return null
  }
  return { status: 'cancelled' }
}

export async function sendChat(
  message: string,
  history: ChatTurn[],
): Promise<ChatApiResponse> {
  return postJson(CHAT_ENDPOINT, { message, history }, asChatResponse)
}

/**
 * Confirm one waiting proposal, which writes its research file.
 *
 * The request only names the proposal: its content, file name and sources all
 * come from the backend's own pending store, so nothing the browser sends can
 * decide what ends up on disk.
 */
export async function saveResearch(saveId: string): Promise<SaveConfirmation> {
  return postJson(
    SAVE_ENDPOINT,
    { save_id: saveId },
    asSaveConfirmation,
    { timeoutMs: DECISION_TIMEOUT_MS, statusMessages: SAVE_STATUS_MESSAGES },
  )
}

/** Drop one waiting proposal, so it can never be written any more. */
export async function cancelResearch(saveId: string): Promise<CancelConfirmation> {
  return postJson(
    SAVE_CANCEL_ENDPOINT,
    { save_id: saveId },
    asCancelConfirmation,
    { timeoutMs: DECISION_TIMEOUT_MS, statusMessages: CANCEL_STATUS_MESSAGES },
  )
}
