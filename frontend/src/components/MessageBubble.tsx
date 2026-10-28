import type { ChatMessage } from '../types'

interface MessageBubbleProps {
  message: ChatMessage
  index: number
  /** Present only for the newest failed turn: renders the retry control. */
  onRetry?: () => void
}

const ROLE_LABELS: Record<ChatMessage['role'], string> = {
  user: 'کاربر',
  assistant: 'دستیار',
}

/** One turn of the transcript, tagged with a mono role label instead of an avatar. */
function MessageBubble({ message, index, onRetry }: MessageBubbleProps) {
  const tag = String(index + 1).padStart(2, '0')

  return (
    <article
      className={`msg msg--${message.role}${message.failed ? ' msg--failed' : ''}`}
    >
      <header className="msg__meta">
        <span className="msg__index">#{tag}</span>
        <span className="msg__role">{ROLE_LABELS[message.role]}</span>
        {message.failed && <span className="msg__flag">ناموفق</span>}
      </header>
      <div className="msg__body">{message.content}</div>
      {onRetry && (
        <button type="button" className="msg__retry" onClick={onRetry}>
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M4 12a8 8 0 1 1 2.34 5.66M4 20v-4.5h4.5"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span>تلاش مجدد</span>
        </button>
      )}
    </article>
  )
}

export default MessageBubble
