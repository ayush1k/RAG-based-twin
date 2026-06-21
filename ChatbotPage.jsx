// ChatbotPage.jsx
//
// A standalone, premium "talk to me" page for your portfolio. Drop this into
// your routes (e.g. /chat or /ask-me) and point BACKEND_URL at your
// deployed FastAPI backend.
//
// If you're on Next.js App Router, add `"use client"` as the very
// first line of this file.

import { useState, useRef, useEffect } from "react";

// TODO: replace with your deployed backend URL
const BACKEND_URL = "http://localhost:8000";

const STARTER_QUESTIONS = [
  "What are you working on right now?",
  "Tell me about your GenAI project",
  "What's your tech stack?",
  "What kind of role are you looking for?",
];

export default function ChatbotPage() {
  const [messages, setMessages] = useState([]); // { role: 'user' | 'assistant', content }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState("dark"); // 'dark' | 'light'
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage(text) {
    const question = text.trim();
    if (!question || loading) return;

    setError(null);
    setInput("");
    const nextMessages = [...messages, { role: "user", content: question }];
    setMessages(nextMessages);
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          history: messages, // history before this turn
        }),
      });

      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();

      setMessages([...nextMessages, { role: "assistant", content: data.answer }]);
    } catch (err) {
      setError("Couldn't reach the bot just now — try again in a moment.");
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendMessage(input);
  }

  function clearHistory() {
    setMessages([]);
    setError(null);
  }

  // A safe, lightweight markdown inline parser for bold, links, code, and raw URLs
  function parseMessageContent(text) {
    if (!text) return "";

    // Regex explanation:
    // 1. Markdown Links: \[[^\]]+\]\(https?:\/\/[^\s)]+\)
    // 2. Bold text: \*\*([^*]+)\*\*
    // 3. Inline code: `([^`]+)`
    // 4. Raw URLs: https?:\/\/[^\s]+
    const tokenRegex = /(\[.*?\]\(https?:\/\/.*?\)|\*\*.*?\*\*|`.*?`|https?:\/\/[^\s\n]+)/g;
    const parts = text.split(tokenRegex);

    return parts.map((part, index) => {
      if (part.startsWith("[") && part.includes("](")) {
        const match = part.match(/\[(.*?)\]\((.*?)\)/);
        if (match) {
          return (
            <a
              key={index}
              href={match[2]}
              target="_blank"
              rel="noopener noreferrer"
              className="chat-page__link"
            >
              {match[1]}
            </a>
          );
        }
      }
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={index} className="chat-page__bold">{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={index} className="chat-page__code">{part.slice(1, -1)}</code>;
      }
      if (part.startsWith("http://") || part.startsWith("https://")) {
        return (
          <a
            key={index}
            href={part}
            target="_blank"
            rel="noopener noreferrer"
            className="chat-page__link"
          >
            {part}
          </a>
        );
      }
      return part;
    });
  }

  return (
    <div className={`chat-page-container ${theme === "dark" ? "theme-dark" : "theme-light"}`}>
      <style>{`
        .chat-page-container {
          --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          min-height: 100vh;
          background: var(--bg);
          color: var(--ink);
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          transition: var(--transition);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          box-sizing: border-box;
        }

        /* Dark Theme Tokens */
        .theme-dark {
          --bg: #0b0f19;
          --card-bg: rgba(22, 30, 49, 0.7);
          --accent: #10b981;
          --accent-soft: rgba(16, 185, 129, 0.12);
          --accent-hover: #059669;
          --ink: #f8fafc;
          --ink-soft: #94a3b8;
          --border: rgba(255, 255, 255, 0.08);
          --shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
          --bubble-user: linear-gradient(135deg, #10b981 0%, #059669 100%);
          --bubble-user-text: #ffffff;
          --bubble-assistant: #1e293b;
          --bubble-assistant-text: #e2e8f0;
          --input-bg: #111827;
        }

        /* Light Theme Tokens */
        .theme-light {
          --bg: #f3f4f6;
          --card-bg: rgba(255, 255, 255, 0.95);
          --accent: #059669;
          --accent-soft: rgba(5, 150, 105, 0.08);
          --accent-hover: #047857;
          --ink: #0f172a;
          --ink-soft: #475569;
          --border: rgba(15, 23, 42, 0.08);
          --shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.1);
          --bubble-user: linear-gradient(135deg, #059669 0%, #047857 100%);
          --bubble-user-text: #ffffff;
          --bubble-assistant: #f1f5f9;
          --bubble-assistant-text: #1e293b;
          --input-bg: #ffffff;
        }

        .chat-page {
          width: 100%;
          max-width: 720px;
          background: var(--card-bg);
          border: 1px solid var(--border);
          border-radius: 24px;
          box-shadow: var(--shadow);
          backdrop-filter: blur(16px);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          transition: var(--transition);
        }

        .chat-page__top-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 24px;
          border-bottom: 1px solid var(--border);
        }

        .chat-page__profile {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .chat-page__avatar {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 700;
          font-size: 16px;
          border: 2px solid var(--border);
        }

        .chat-page__status-container {
          display: flex;
          flex-direction: column;
        }

        .chat-page__name {
          font-size: 16px;
          font-weight: 600;
          margin: 0;
        }

        .chat-page__status {
          font-size: 12px;
          color: var(--ink-soft);
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 2px;
        }

        .chat-page__status-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background-color: var(--accent);
          box-shadow: 0 0 8px var(--accent);
          display: inline-block;
        }

        .chat-page__actions {
          display: flex;
          gap: 10px;
        }

        .chat-page__btn-icon {
          background: transparent;
          border: 1px solid var(--border);
          color: var(--ink);
          width: 38px;
          height: 38px;
          border-radius: 10px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: var(--transition);
        }

        .chat-page__btn-icon:hover {
          background: var(--accent-soft);
          border-color: var(--accent);
          color: var(--accent);
        }

        .chat-page__intro {
          padding: 24px 24px 16px;
          text-align: left;
        }

        .chat-page__title {
          font-size: 24px;
          font-weight: 700;
          margin: 0 0 8px;
          letter-spacing: -0.02em;
        }

        .chat-page__subtitle {
          font-size: 14.5px;
          color: var(--ink-soft);
          line-height: 1.5;
          margin: 0;
        }

        .chat-page__log {
          height: 400px;
          overflow-y: auto;
          padding: 20px 24px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          border-top: 1px solid var(--border);
          scrollbar-width: thin;
          scrollbar-color: var(--border) transparent;
        }

        .chat-page__log::-webkit-scrollbar {
          width: 6px;
        }

        .chat-page__log::-webkit-scrollbar-thumb {
          background-color: var(--border);
          border-radius: 99px;
        }

        .chat-page__empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          text-align: center;
          gap: 16px;
          margin: auto;
          max-width: 440px;
          animation: fadeIn 0.4s ease-out;
        }

        .chat-page__empty-text {
          font-size: 14px;
          color: var(--ink-soft);
          margin: 0;
        }

        .chat-page__chips {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          justify-content: center;
        }

        .chat-page__chip {
          font-size: 13px;
          padding: 10px 16px;
          border-radius: 99px;
          border: 1px solid var(--border);
          background: var(--input-bg);
          color: var(--ink);
          cursor: pointer;
          transition: var(--transition);
          box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }

        .chat-page__chip:hover {
          border-color: var(--accent);
          background: var(--accent-soft);
          color: var(--accent);
          transform: translateY(-1px);
        }

        .chat-page__bubble-wrapper {
          display: flex;
          flex-direction: column;
          max-width: 80%;
          animation: slideUp 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .chat-page__bubble-wrapper--user {
          align-self: flex-end;
          align-items: flex-end;
        }

        .chat-page__bubble-wrapper--assistant {
          align-self: flex-start;
          align-items: flex-start;
        }

        .chat-page__bubble {
          padding: 12px 18px;
          border-radius: 18px;
          font-size: 14.5px;
          line-height: 1.6;
          white-space: pre-wrap;
          box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }

        .chat-page__bubble--user {
          background: var(--bubble-user);
          color: var(--bubble-user-text);
          border-bottom-right-radius: 4px;
        }

        .chat-page__bubble--assistant {
          background: var(--bubble-assistant);
          color: var(--bubble-assistant-text);
          border-bottom-left-radius: 4px;
          border: 1px solid var(--border);
        }

        .chat-page__link {
          color: var(--accent);
          text-decoration: underline;
          font-weight: 500;
          transition: var(--transition);
        }

        .chat-page__link:hover {
          color: var(--accent-hover);
        }

        .chat-page__bold {
          font-weight: 600;
        }

        .chat-page__code {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          background: rgba(0,0,0,0.15);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 13px;
        }

        .chat-page__typing {
          align-self: flex-start;
          display: flex;
          align-items: center;
          gap: 5px;
          padding: 14px 20px;
          border-radius: 18px;
          border-bottom-left-radius: 4px;
          background: var(--bubble-assistant);
          border: 1px solid var(--border);
          animation: slideUp 0.3s ease-out;
        }

        .chat-page__typing span {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: var(--accent);
          opacity: 0.4;
          animation: chatTypingPulse 1.2s ease-in-out infinite;
        }

        .chat-page__typing span:nth-child(2) { animation-delay: 0.2s; }
        .chat-page__typing span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes chatTypingPulse {
          0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
          30% { opacity: 1; transform: translateY(-3px); }
        }

        @keyframes slideUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        .chat-page__error {
          font-size: 13.5px;
          color: #ef4444;
          margin: 12px 24px 0;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .chat-page__form-container {
          padding: 16px 24px 24px;
          border-top: 1px solid var(--border);
        }

        .chat-page__form {
          display: flex;
          gap: 10px;
        }

        .chat-page__input {
          flex: 1;
          padding: 14px 18px;
          border-radius: 14px;
          border: 1px solid var(--border);
          background: var(--input-bg);
          color: var(--ink);
          font-size: 14.5px;
          font-family: inherit;
          outline: none;
          transition: var(--transition);
          box-shadow: inset 0 2px 4px rgba(0,0,0,0.01);
        }

        .chat-page__input:focus {
          border-color: var(--accent);
          box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .chat-page__send {
          padding: 0 24px;
          border-radius: 14px;
          border: none;
          background: var(--accent);
          color: #ffffff;
          font-size: 14.5px;
          font-weight: 600;
          cursor: pointer;
          transition: var(--transition);
        }

        .chat-page__send:hover:not(:disabled) {
          background: var(--accent-hover);
          transform: translateY(-1px);
        }

        .chat-page__send:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .chat-page__send:focus-visible,
        .chat-page__btn-icon:focus-visible,
        .chat-page__chip:focus-visible,
        .chat-page__input:focus-visible {
          outline: 2px solid var(--accent);
          outline-offset: 2px;
        }
      `}</style>

      <div className="chat-page">
        {/* Top Header Bar */}
        <div className="chat-page__top-bar">
          <div className="chat-page__profile">
            <div className="chat-page__avatar">A</div>
            <div className="chat-page__status-container">
              <h2 className="chat-page__name">Ayush Clone</h2>
              <span className="chat-page__status" aria-label="Status Online">
                <span className="chat-page__status-dot" /> Grounded RAG Bot
              </span>
            </div>
          </div>
          <div className="chat-page__actions">
            {/* Theme Toggle Button */}
            <button
              className="chat-page__btn-icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.072.072a5 5 0 01-7 0z" />
                </svg>
              ) : (
                <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
            {/* Clear Chat Button */}
            <button
              className="chat-page__btn-icon"
              onClick={clearHistory}
              title="Clear chat history"
              aria-label="Clear chat"
              disabled={messages.length === 0}
              style={{ opacity: messages.length === 0 ? 0.4 : 1 }}
            >
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Intro Section */}
        <div className="chat-page__intro">
          <h1 className="chat-page__title">Ask me about my background</h1>
          <p className="chat-page__subtitle">
            I've built this interactive RAG twin to answer questions about my skills,
            academic journey (M.Tech in AI & Data Science), research experience, and projects.
          </p>
        </div>

        {/* Chat Message Area */}
        <div className="chat-page__log" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat-page__empty">
              <p className="chat-page__empty-text">Select a prompt below or type your own question to start chatting:</p>
              <div className="chat-page__chips">
                {STARTER_QUESTIONS.map((q) => (
                  <button key={q} className="chat-page__chip" onClick={() => sendMessage(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={`chat-page__bubble-wrapper chat-page__bubble-wrapper--${m.role}`}
            >
              <div className={`chat-page__bubble chat-page__bubble--${m.role}`}>
                {parseMessageContent(m.content)}
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-page__typing" aria-label="Thinking">
              <span />
              <span />
              <span />
            </div>
          )}
        </div>

        {/* Error Indicator */}
        {error && (
          <div className="chat-page__error">
            <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Chat Input form */}
        <div className="chat-page__form-container">
          <form className="chat-page__form" onSubmit={handleSubmit}>
            <input
              className="chat-page__input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about my resume, skills, or M.Tech research..."
              aria-label="Your query"
            />
            <button className="chat-page__send" type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

