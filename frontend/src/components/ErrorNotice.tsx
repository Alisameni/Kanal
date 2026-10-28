interface ErrorNoticeProps {
  message: string
  onDismiss: () => void
}

/** A plain-language failure message; the text is already prepared by the API layer. */
function ErrorNotice({ message, onDismiss }: ErrorNoticeProps) {
  return (
    <div className="error" role="alert">
      <span className="error__tag">خطا</span>
      <p className="error__text">{message}</p>
      <button
        type="button"
        className="error__close"
        onClick={onDismiss}
        aria-label="بستن هشدار"
      >
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M5 5l14 14M19 5L5 19"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </div>
  )
}

export default ErrorNotice
