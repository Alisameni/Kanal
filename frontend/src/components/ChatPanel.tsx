import type { ChatStatus, SaveProposal } from '../hooks/useChat'
import type { ChatMessage } from '../types'
import Composer from './Composer'
import ErrorNotice from './ErrorNotice'
import MessageList from './MessageList'
import SaveResearchCard from './SaveResearchCard'
import TypingIndicator from './TypingIndicator'

interface ChatPanelProps {
  messages: ChatMessage[]
  status: ChatStatus
  error: string | null
  /** The save proposal of the newest reply, or null when there is none. */
  proposal: SaveProposal | null
  onSend: (text: string) => void
  onRetry: (id: string) => void
  onDismissError: () => void
  onConfirmSave: () => void
  onCancelSave: () => void
  onDismissProposal: () => void
}

/** Shown before the first message: a short briefing on how the channel works. */
function Standby() {
  return (
    <section className="standby">
      <p className="standby__kicker">channel ready</p>
      <h2 className="standby__title">کانال باز است</h2>
      <p className="standby__text">
        پیام خود را بنویسید؛ هر درخواست همراه تاریخچهٔ همین مکالمه فرستاده می‌شود و پاسخ
        دستیار همان‌جا نمایش داده خواهد شد.
      </p>
      <ul className="standby__list">
        <li>RTL</li>
        <li>POST /chat</li>
        <li>بدون ذخیره‌سازی</li>
      </ul>
    </section>
  )
}

function ChatPanel({
  messages,
  status,
  error,
  proposal,
  onSend,
  onRetry,
  onDismissError,
  onConfirmSave,
  onCancelSave,
  onDismissProposal,
}: ChatPanelProps) {
  const busy = status === 'pending'

  return (
    <main className="chat">
      <header className="chat__header">
        <div className="chat__heading">
          <h1 className="chat__title">گفت‌وگو</h1>
          <span className="chat__subtitle">CH-01 · KANAL</span>
        </div>
        <span className="chat__count">{messages.length.toLocaleString('fa-IR')} پیام</span>
      </header>

      <div className="chat__thread">
        {messages.length === 0 ? (
          <Standby />
        ) : (
          <MessageList messages={messages} busy={busy} onRetry={onRetry} />
        )}
        {/*
         * The card follows the reply it belongs to. `key` is the proposal's id,
         * so a new proposal remounts the card (collapsed, scrolled into view)
         * while a state change of the same proposal only re-renders it.
         */}
        {proposal && (
          <SaveResearchCard
            key={proposal.saveId}
            proposal={proposal}
            onConfirm={onConfirmSave}
            onCancel={onCancelSave}
            onDismiss={onDismissProposal}
          />
        )}
        {busy && <TypingIndicator />}
        {error && <ErrorNotice message={error} onDismiss={onDismissError} />}
      </div>

      <footer className="chat__footer">
        <Composer onSend={onSend} busy={busy} />
        <p className="chat__hint">Enter ارسال · Shift+Enter خط جدید</p>
      </footer>
    </main>
  )
}

export default ChatPanel
