import { useEffect, useRef, useState } from 'react'

import { isProposalBusy, isProposalOpen } from '../hooks/useChat'
import type { SaveProposal, SaveProposalStatus } from '../hooks/useChat'
import { isSafeHttpUrl } from '../lib/url'
import MarkdownPreview from './MarkdownPreview'

interface SaveResearchCardProps {
  proposal: SaveProposal
  /** Writes the research file of this proposal. Never called automatically. */
  onConfirm: () => void
  /** Drops the proposal, so it can never be written. */
  onCancel: () => void
  /** Hides a proposal that is already decided. */
  onDismiss: () => void
}

/** The badge in the card's corner: one short word per state. */
const STATUS_LABELS: Record<SaveProposalStatus, string> = {
  pending: 'در انتظار تأیید',
  saving: 'در حال ذخیره',
  saved: 'ذخیره شد',
  cancelling: 'در حال لغو',
  cancelled: 'لغو شد',
  error: 'خطا',
}

/**
 * The sentence that says what has and has not happened yet.
 *
 * It is the most important line of the card: the document is still only a
 * proposal, and nothing exists on disk until the user confirms it.
 */
function statusNote(proposal: SaveProposal): string {
  switch (proposal.status) {
    case 'pending':
      return 'این فقط یک پیشنهاد است؛ تا شما تأیید نکنید هیچ فایلی ذخیره نمی‌شود.'
    case 'saving':
      return 'در حال ذخیرهٔ فایل؛ نتیجه به‌زودی همین‌جا نمایش داده می‌شود.'
    case 'saved':
      return `فایل «${proposal.filename}» ذخیره شد.`
    case 'cancelling':
      return 'در حال لغو پیشنهاد؛ نتیجه به‌زودی همین‌جا نمایش داده می‌شود.'
    case 'cancelled':
      return 'این پیشنهاد لغو شد و هیچ فایلی ذخیره نخواهد شد.'
    case 'error':
      return proposal.retryable
        ? 'هنوز هیچ فایلی ذخیره نشده است؛ می‌توانید دوباره تلاش کنید.'
        : 'این پیشنهاد دیگر قابل استفاده نیست و ذخیره نمی‌شود.'
  }
}

/** The label of the primary button, which doubles as its progress report. */
function confirmLabel(proposal: SaveProposal): string {
  if (proposal.status === 'saving') {
    return 'در حال ذخیره...'
  }
  if (proposal.status === 'error') {
    return 'تلاش مجدد برای ذخیره'
  }
  return 'تأیید و ذخیره'
}

/**
 * The review card of one save proposal.
 *
 * It is purely presentational: every state and every request lives in `useChat`,
 * and the only way a file is ever written is the confirm button calling
 * `onConfirm`. The card itself can therefore never save anything, and it never
 * renders backend text as markup: the title and summary are plain child text,
 * the preview goes through react-markdown with raw HTML disabled, and a source
 * is only listed when its link is a real http/https URL.
 */
function SaveResearchCard({
  proposal,
  onConfirm,
  onCancel,
  onDismiss,
}: SaveResearchCardProps) {
  const [showSources, setShowSources] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const cardRef = useRef<HTMLElement>(null)

  // Only links the UI is allowed to open are listed, and the count follows that
  // list instead of whatever the payload claimed.
  const sources = proposal.sources.filter((source) => isSafeHttpUrl(source.url))
  const preview = proposal.preview
  const open = isProposalOpen(proposal)
  const busy = isProposalBusy(proposal)

  // The card replaces itself in place, so bring it into view when it appears;
  // otherwise a long reply could push the decision below the fold.
  useEffect(() => {
    cardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [])

  return (
    <section
      ref={cardRef}
      className={`save-card save-card--${proposal.status}`}
      aria-label="پیشنهاد ذخیرهٔ تحقیق"
    >
      <header className="save-card__head">
        <span className="save-card__kicker">پیشنهاد ذخیرهٔ تحقیق</span>
        <span className="save-card__status">{STATUS_LABELS[proposal.status]}</span>
      </header>

      <h3 className="save-card__title">{proposal.title || 'تحقیق بدون عنوان'}</h3>

      {proposal.summary && (
        <p className="save-card__summary">{proposal.summary}</p>
      )}

      <p className="save-card__note" role="status" aria-live="polite">
        {statusNote(proposal)}
      </p>

      {proposal.status === 'error' && proposal.error && (
        <p className="save-card__error" role="alert">
          {proposal.error}
        </p>
      )}

      <dl className="save-card__meta">
        <div className="save-card__meta-row">
          <dt>نام فایل</dt>
          <dd>{proposal.filename || '—'}</dd>
        </div>
        <div className="save-card__meta-row">
          <dt>منابع</dt>
          <dd>{sources.length.toLocaleString('fa-IR')} مورد</dd>
        </div>
      </dl>

      {(sources.length > 0 || preview !== null) && (
        <div className="save-card__tools">
          {sources.length > 0 && (
            <button
              type="button"
              className="save-card__toggle"
              aria-expanded={showSources}
              onClick={() => setShowSources((value) => !value)}
            >
              <span>
                {showSources
                  ? 'بستن منابع'
                  : `نمایش منابع (${sources.length.toLocaleString('fa-IR')})`}
              </span>
            </button>
          )}

          {preview !== null && (
            <button
              type="button"
              className="save-card__toggle"
              aria-expanded={showPreview}
              onClick={() => setShowPreview((value) => !value)}
            >
              <span>
                {showPreview ? 'بستن پیش‌نمایش' : 'نمایش پیش‌نمایش Markdown'}
              </span>
            </button>
          )}
        </div>
      )}

      {showSources && sources.length > 0 && (
        <ul className="save-card__sources">
          {sources.map((source) => (
            <li key={source.url}>
              <a href={source.url} target="_blank" rel="noreferrer noopener">
                {source.title}
              </a>
            </li>
          ))}
        </ul>
      )}

      {showPreview && preview !== null && (
        <div className="save-card__preview">
          {/*
           * MarkdownPreview renders React elements only: raw HTML is skipped and
           * every URL is reduced to http/https, so neither markup nor a
           * dangerous link can reach the page.
           */}
          <MarkdownPreview source={preview} />
        </div>
      )}

      <footer className="save-card__actions">
        {open ? (
          <>
            <button
              type="button"
              className="save-card__confirm"
              onClick={onConfirm}
              disabled={busy}
            >
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M12 4v10m-4-4 4 4 4-4M5 20h14"
                  stroke="currentColor"
                  strokeWidth="1.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span>{confirmLabel(proposal)}</span>
            </button>

            <button
              type="button"
              className="save-card__cancel"
              onClick={onCancel}
              disabled={busy}
            >
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M5 5l14 14M19 5L5 19"
                  stroke="currentColor"
                  strokeWidth="1.9"
                  strokeLinecap="round"
                />
              </svg>
              <span>
                {proposal.status === 'cancelling' ? 'در حال لغو...' : 'لغو'}
              </span>
            </button>
          </>
        ) : (
          <button
            type="button"
            className="save-card__dismiss"
            onClick={onDismiss}
          >
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M5 5l14 14M19 5L5 19"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
            </svg>
            <span>بستن کارت</span>
          </button>
        )}
      </footer>
    </section>
  )
}

export default SaveResearchCard