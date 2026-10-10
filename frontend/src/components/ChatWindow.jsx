import { useEffect, useRef, useState } from "react";
import { sendChatMessage } from "../services/api";
import Message from "./Message";
import InputBox from "./InputBox";

const starterMessage = {
  role: "assistant",
  content:
    "Hi! Tell me what product you're looking for, your budget, and the features that matter most. I'll help you compare options from the product catalog.",
};

export default function ChatWindow() {
  const [messages, setMessages] = useState([starterMessage]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(text) {
    const history = messages
      .filter((item) => item.role === "user" || item.role === "assistant")
      .filter((item) => !item.data)
      .map((item) => ({ role: item.role, content: item.content }));

    setMessages((previous) => [...previous, { role: "user", content: text }]);
    setLoading(true);
    setError("");

    try {
      const data = await sendChatMessage(text, history);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.assistant_message,
          data,
        },
      ]);
    } catch (err) {
      setError(err.message || "Unable to reach the recommendation service.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="chat-panel">
      <header className="chat-panel__header">
        <div className="brand-mark">PF</div>
        <div>
          <h2>ProductFinder AI</h2>
          <p>Conversational product recommendations</p>
        </div>
        <span className="connection-status">Chat</span>
      </header>

      <div className="chat-messages" aria-live="polite">
        {messages.map((message, index) => (
          <Message key={`${message.role}-${index}`} message={message} />
        ))}

        {loading && (
          <div className="typing-indicator" role="status">
            <span />
            <span />
            <span />
            <span className="sr-only">Preparing your recommendation</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="chat-error" role="alert">
          {error}
        </div>
      )}

      <InputBox onSend={handleSend} disabled={loading} />
    </section>
  );
}
