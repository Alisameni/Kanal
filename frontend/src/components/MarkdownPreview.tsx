import type { ComponentProps } from 'react'
import Markdown from 'react-markdown'
import type { Components } from 'react-markdown'

import { isSafeHttpUrl, safeUrlTransform } from '../lib/url'

interface MarkdownPreviewProps {
  /** Markdown text from the backend. Shown as data, never as markup. */
  source: string
}

/**
 * A link inside the rendered Markdown.
 *
 * The renderer has already reduced every URL to http/https (`safeUrlTransform`),
 * and this component refuses anything else a second time, so a link can never
 * come out as `javascript:` or as a same-page navigation. A refused link keeps
 * its text but loses its address.
 */
function MarkdownLink({ href, children }: ComponentProps<'a'>) {
  if (!href || !isSafeHttpUrl(href)) {
    return <span>{children}</span>
  }

  return (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  )
}

/**
 * An image inside the rendered Markdown.
 *
 * Only an http/https source is drawn at all; anything else (including the empty
 * string the renderer produces for a refused URL) renders nothing, so a browser
 * never asks for a half-built address. Remote images are loaded lazily and
 * without a referrer.
 */
function MarkdownImage({ src, alt }: ComponentProps<'img'>) {
  if (!src || !isSafeHttpUrl(src)) {
    return null
  }

  return (
    <img
      src={src}
      alt={alt ?? ''}
      loading="lazy"
      referrerPolicy="no-referrer"
    />
  )
}

/** Defined once, so react-markdown does not receive a new map on every render. */
const MARKDOWN_COMPONENTS: Components = { a: MarkdownLink, img: MarkdownImage }

/**
 * The only place Markdown from the backend is turned into elements.
 *
 * It is deliberately narrow, because the document is untrusted text:
 *
 * - react-markdown builds React elements; there is no `innerHTML` and no
 *   `dangerouslySetInnerHTML` anywhere in this path, so nothing can execute.
 * - `skipHtml` drops raw HTML (`<script>`, `<img onerror=…>`) instead of
 *   rendering or escaping it into something the browser would parse.
 * - `urlTransform` keeps only absolute http/https URLs, so `javascript:`,
 *   `data:` and relative addresses never reach an attribute.
 * - Links open in a new tab without a referrer or an opener handle.
 */
function MarkdownPreview({ source }: MarkdownPreviewProps) {
  return (
    <Markdown
      skipHtml
      urlTransform={safeUrlTransform}
      components={MARKDOWN_COMPONENTS}
    >
      {source}
    </Markdown>
  )
}

export default MarkdownPreview