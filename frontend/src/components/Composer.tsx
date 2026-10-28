import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent, KeyboardEvent } from 'react'

import { MAX_MESSAGE_LENGTH } from '../api/chat'

interface ComposerProps {
  onSend: (text: string) => void
  busy: boolean
}

/** Matches `max-height` on `.composer__input` in app.css. */
const MAX_HEIGHT = 180

function Composer({ onSend, busy }: ComposerProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow with the content, up to MAX_HEIGHT, then scroll inside the textarea.
  useEffect(() => {
    const node = textareaRef.current
    if (!node) {
      return
    }
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, MAX_HEIGHT)}px`
  }, [value])

  const submit = () => {
    const text = value.trim()
    if (!text || busy) {
      return
    }
    onSend(text)
    setValue('')
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    submit()
  }

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(event.target.value)
  }

  // Enter sends; Shift+Enter keeps the newline.
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const canSend = value.trim().length > 0 && !busy

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        ref={textareaRef}
        className="composer__input"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="پیام خود را بنویسید…"
        aria-label="متن پیام"
        maxLength={MAX_MESSAGE_LENGTH}
        rows={1}
      />
      <button type="submit" className="composer__send" disabled={!canSend}>
        <span>ارسال</span>
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M20 12H5m5-6-6 6 6 6"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </form>
  )
}

export default Composer
