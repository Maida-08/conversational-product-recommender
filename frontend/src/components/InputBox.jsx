import { useState } from "react";

export default function InputBox({ onSend, disabled }) {
  const [message, setMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = message.trim();

    if (!trimmed || disabled) return;

    setMessage("");
    await onSend(trimmed);
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="chat-message">
        Describe the product you want
      </label>

      <textarea
        id="chat-message"
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        placeholder="Tell me what you're looking for, your budget, or must-have features..."
        rows={2}
        maxLength={1000}
        disabled={disabled}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />

      <div className="chat-input__footer">
        <span>Enter to send · Shift + Enter for a new line</span>
        <button type="submit" disabled={disabled || !message.trim()}>
          {disabled ? "Thinking..." : "Send ↑"}
        </button>
      </div>
    </form>
  );
}
