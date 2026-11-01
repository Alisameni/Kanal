import { useEffect, useState } from 'react'

/**
 * After this long the wait is worth explaining instead of only animating.
 *
 * A search turn really does take this long sometimes: the model searches, reads
 * the results and then writes a long answer, and the backend allows it up to 100s
 * (see TOOL_DEADLINE_SECONDS in main.py). Saying so beats leaving the user with
 * three dots and no idea whether anything is happening.
 */
const SLOW_AFTER_MS = 20_000

/** Shown while the backend call for the current message is still pending. */
function TypingIndicator() {
  const [slow, setSlow] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), SLOW_AFTER_MS)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="typing" role="status" aria-live="polite">
      <span className="typing__dots" aria-hidden="true">
        <span className="typing__dot" />
        <span className="typing__dot" />
        <span className="typing__dot" />
      </span>
      <span className="typing__label">
        {slow
          ? 'در حال جستوجو و تحلیل نتایج؛ پاسخهای بلند کمی طول میکشد'
          : 'در حال پردازش پاسخ'}
      </span>
    </div>
  )
}

export default TypingIndicator
