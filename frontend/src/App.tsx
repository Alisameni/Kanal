import ChatPanel from './components/ChatPanel'
import Sidebar from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useTheme } from './hooks/useTheme'

function App() {
  const {
    messages,
    status,
    error,
    connection,
    sendMessage,
    retryMessage,
    clearChat,
    dismissError,
  } = useChat()
  const { theme, setTheme } = useTheme()

  return (
    <div className="app">
      <Sidebar
        theme={theme}
        connection={connection}
        busy={status === 'pending'}
        canClear={messages.length > 0}
        onSelectTheme={setTheme}
        onClear={clearChat}
      />
      <ChatPanel
        messages={messages}
        status={status}
        error={error}
        onSend={sendMessage}
        onRetry={retryMessage}
        onDismissError={dismissError}
      />
    </div>
  )
}

export default App
