import { useEffect, useRef } from 'react'

import type { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

interface MessageListProps {
  messages: ChatMessage[]
  busy: boolean
  onRetry: (id: string) => void
}

function MessageList({ messages, busy, onRetry }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  // Keep the newest turn in view whenever the transcript grows.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <>
      {messages.map((message, index) => {
        // Only the newest turn can be retried in place; an older failed message
        // stays exactly where it is.
        const canRetry = message.failed && index === messages.length - 1 && !busy
        return (
          <MessageBubble
            key={message.id}
            message={message}
            index={index}
            onRetry={canRetry ? () => onRetry(message.id) : undefined}
          />
        )
      })}
      <div ref={endRef} />
    </>
  )
}

export default MessageList
