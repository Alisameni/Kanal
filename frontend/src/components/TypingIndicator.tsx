/** Shown while the backend call for the current message is still pending. */
function TypingIndicator() {
  return (
    <div className="typing" role="status" aria-live="polite">
      <span className="typing__dots" aria-hidden="true">
        <span className="typing__dot" />
        <span className="typing__dot" />
        <span className="typing__dot" />
      </span>
      <span className="typing__label">در حال پردازش پاسخ</span>
    </div>
  )
}

export default TypingIndicator
