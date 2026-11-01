/**
 * Link safety helpers.
 *
 * Every link the UI shows comes from the backend, and the backend is the only
 * place that can validate what it prepared (see `public_sources` in server.py).
 * These helpers are the frontend's own guard, so nothing is trusted twice: only
 * an absolute `http`/`https` URL that actually names a host is ever treated as a
 * link. A `javascript:`, `data:`, `file:` or relative URL can therefore neither
 * become an anchor nor survive the Markdown renderer.
 */

/** True only for an absolute http/https URL with a host. */
export function isSafeHttpUrl(value: string): boolean {
  if (typeof value !== 'string') {
    return false
  }

  let url: URL
  try {
    url = new URL(value)
  } catch {
    return false
  }

  return (
    (url.protocol === 'http:' || url.protocol === 'https:') &&
    url.hostname.length > 0
  )
}

/**
 * The `urlTransform` handed to the Markdown renderer.
 *
 * react-markdown calls it for every URL it is about to emit (links, images,
 * anything with an address) and omits the attribute entirely for an empty
 * answer, so an unsafe URL is dropped instead of being rendered.
 */
export function safeUrlTransform(value: string): string {
  return isSafeHttpUrl(value) ? value : ''
}
