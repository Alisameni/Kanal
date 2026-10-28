import type { ChatApiResponse, ChatTurn } from '../types'

/**
 * The single place the UI talks to the backend.
 *
 * Only `POST /chat` is used, and it is requested with a relative path so the
 * Vite dev/preview proxy (see vite.config.ts) can forward it to FastAPI without
 * any CORS setup. No API key, base URL or `.env` value ever reaches the browser.
 */

const CHAT_ENDPOINT = '/chat'

/**
 * Sits above the backend's worst case so the browser never aborts a request the
 * server is still legitimately working on.
 *
 * The OpenAI client in main.py uses a 20s per-operation timeout with one retry
 * (`max_retries=1`), so a slow provider can take roughly `2 * 20s + backoff`
 * (~41s) before the server gives up. That 20s is a per-operation budget, not a
 * whole-request deadline, and a provider `Retry-After` header can stretch it
 * further, so this 45s ceiling is a safety margin rather than an exact bound.
 *
 * Aborting here only stops the browser from waiting; it does not guarantee the
 * backend stops working on the request.
 */
const REQUEST_TIMEOUT_MS = 45_000

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
  504: 'پاسخ سرور بیش از حد طول کشید.',
}

/** An error the UI can show verbatim. `status === null` means "never reached the server". */
export class ChatApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ChatApiError'
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
function describeFailure(status: number): string {
  return STATUS_MESSAGES[status] ?? `درخواست با خطای ${status} ناموفق بود.`
}

export async function sendChat(
  message: string,
  history: ChatTurn[],
): Promise<ChatApiResponse> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  // The timer is cleared once, in the outermost `finally`, so it also protects
  // the body read: `fetch()` resolving only means the response *headers* have
  // arrived, not that the whole body has been read and parsed.
  try {
    let response: Response
    try {
      response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history }),
        signal: controller.signal,
      })
    } catch (cause) {
      throw new ChatApiError(
        isAbortError(cause) ? TIMEOUT_MESSAGE : NETWORK_MESSAGE,
      )
    }

    if (!response.ok) {
      throw new ChatApiError(describeFailure(response.status), response.status)
    }

    let data: ChatApiResponse
    try {
      data = (await response.json()) as ChatApiResponse
    } catch (cause) {
      throw new ChatApiError(
        isAbortError(cause) ? TIMEOUT_MESSAGE : INVALID_PAYLOAD_MESSAGE,
        response.status,
      )
    }

    if (!data || typeof data.reply !== 'string') {
      throw new ChatApiError(INVALID_PAYLOAD_MESSAGE, response.status)
    }

    return data
  } finally {
    clearTimeout(timer)
  }
}
