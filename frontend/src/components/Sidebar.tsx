import type { ConnectionState } from '../hooks/useChat'
import type { Theme } from '../hooks/useTheme'
import ThemeToggle from './ThemeToggle'

interface SidebarProps {
  theme: Theme
  connection: ConnectionState
  busy: boolean
  canClear: boolean
  onSelectTheme: (theme: Theme) => void
  onClear: () => void
}

const CONNECTION_LABELS: Record<ConnectionState, string> = {
  unknown: 'در انتظار اتصال',
  online: 'متصل به سرور',
  offline: 'ارتباط قطع است',
}

function Sidebar({
  theme,
  connection,
  busy,
  canClear,
  onSelectTheme,
  onClear,
}: SidebarProps) {
  // While a request is in flight the LED shows activity, not connectivity.
  const ledState = busy ? 'pending' : connection
  const label = busy ? 'در حال پردازش' : CONNECTION_LABELS[connection]

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="brand__glyph" aria-hidden="true">
          <svg viewBox="0 0 32 32" fill="none">
            <rect x="3" y="3" width="26" height="26" rx="4" stroke="currentColor" strokeWidth="1.2" />
            <path className="rail" d="M10 22V10M22 22V10" strokeWidth="1.8" strokeLinecap="round" />
            <path className="signal" d="M10 16h12" strokeWidth="1.8" strokeLinecap="round" />
            <circle className="core" cx="16" cy="16" r="2.6" />
          </svg>
        </span>
        <span className="brand__text">
          <span className="brand__name">کانال</span>
          <span className="brand__sub">personal chatbot</span>
        </span>
      </div>

      <div className={`status status--${ledState}`} role="status" aria-live="polite">
        <span className="status__led" aria-hidden="true" />
        <span className="status__text">{label}</span>
      </div>

      <p className="sidebar__note">
        گفت‌وگو فقط در همین مرورگر می‌ماند؛ هر پیام همراه تاریخچهٔ مکالمه به سرور فرستاده
        می‌شود و پس از پاسخ، چیزی ذخیره نمی‌گردد.
      </p>

      <div className="sidebar__actions">
        <ThemeToggle theme={theme} onSelect={onSelectTheme} />
        <button
          type="button"
          className="btn-ghost"
          onClick={onClear}
          disabled={!canClear || busy}
        >
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M5 5l14 14M19 5L5 19"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
          <span className="btn-ghost__label">پاک‌کردن مکالمه</span>
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
