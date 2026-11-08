
import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";

import ReactMarkdown from "react-markdown";

import "./App.css";

import {
  api,
  type ChatResponse,
  type Message,
  type Session,
  type Source,
} from "./services/api";

interface DisplayMessage {
  role: "human" | "ai";
  content: string;
  sources?: Source[];
}

function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [question, setQuestion] = useState("");

  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Automatically scroll to the latest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, sending]);

  // Retrieve all conversations.
  async function loadSessions() {
    const data = await api.getSessions();
    setSessions(data);
    return data;
  }

  // Load a conversation and its saved messages.
  async function loadSession(sessionId: string) {
    const session = await api.getSession(sessionId);

    setActiveSessionId(sessionId);

    setMessages(
      session.messages
        .filter((message: Message) => message.role !== "system")
        .map((message: Message) => ({
          role: message.role as "human" | "ai",
          content: message.content,
          sources: message.sources,
        }))
    );

    setError("");
  }

  // Initialize the application.
  useEffect(() => {
    async function initialize() {
      try {
        const data = await loadSessions();

        if (data.length > 0) {
          await loadSession(data[0].session_id);
        }
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to connect to backend."
        );
      } finally {
        setLoading(false);
      }
    }

    initialize();
  }, []);

  // Create a new conversation.
  async function createNewSession() {
    try {
      setError("");

      const result = await api.createSession();

      await loadSessions();
      await loadSession(result.session_id);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to create conversation."
      );
    }
  }

  // Switch between conversations.
  async function handleSelectSession(sessionId: string) {
    try {
      await loadSession(sessionId);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load conversation."
      );
    }
  }

  // Send a question to the backend.
 
async function handleSendMessage() {
  const trimmedQuestion = question.trim();

  if (!trimmedQuestion || !activeSessionId || sending) return;

  const sessionId = activeSessionId;

  setQuestion("");
  setError("");
  setSending(true);

  // Display the user's message immediately.
  setMessages((previous) => [
    ...previous,
    {
      role: "human",
      content: trimmedQuestion,
    },
  ]);

  try {
    const response: ChatResponse = await api.sendMessage(
      sessionId,
      trimmedQuestion
    );

    // Display the assistant's response and its sources.
    setMessages((previous) => [
      ...previous,
      {
        role: "ai",
        content: response.answer,
        sources: response.sources,
      },
    ]);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Unable to send your message."
    );

    // Reload persisted messages, including their sources.
    try {
      const session = await api.getSession(sessionId);

      setMessages(
        session.messages
          .filter((message) => message.role !== "system")
          .map((message) => ({
            role: message.role as "human" | "ai",
            content: message.content,
            sources: message.sources,
          }))
      );
    } catch {
      // Preserve the original error if reloading also fails.
    }
  } finally {
    setSending(false);
  }
}


  // Delete a conversation.
  async function handleDeleteSession(sessionId: string) {
    if (!window.confirm("Delete this conversation?")) {
      return;
    }

    try {
      setError("");

      await api.deleteSession(sessionId);

      const remainingSessions = await loadSessions();

      if (sessionId === activeSessionId) {
        if (remainingSessions.length > 0) {
          await loadSession(remainingSessions[0].session_id);
        } else {
          setActiveSessionId(null);
          setMessages([]);
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to delete conversation."
      );
    }
  }

  // Enter sends; Shift + Enter inserts a new line.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }

  if (loading) {
    return <div className="loading-screen">Loading assistant...</div>;
  }

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>AI Assistant</h2>

          <button
            className="new-chat-button"
            onClick={createNewSession}
            disabled={sending}
          >
            + New Chat
          </button>
        </div>

        <div className="session-list">
          <p className="sidebar-label">CONVERSATIONS</p>

          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`session-item ${
                activeSessionId === session.session_id ? "active" : ""
              }`}
            >
              <button
                className="session-select"
                onClick={() => handleSelectSession(session.session_id)}
                title={session.session_id}
              >
                {session.session_id}
              </button>

              <button
                className="delete-button"
                onClick={() => handleDeleteSession(session.session_id)}
                title="Delete conversation"
                aria-label={`Delete ${session.session_id}`}
                disabled={sending}
              >
                ×
              </button>
            </div>
          ))}

          {sessions.length === 0 && (
            <p className="empty-sessions">No conversations yet.</p>
          )}
        </div>
      </aside>

      {/* Chat */}
      <main className="chat-container">
        <header className="chat-header">
          <div>
            <h1>Session-Based AI Knowledge Assistant</h1>
            <p>Ask questions based on your uploaded documents.</p>
          </div>
        </header>

        <section className="messages-area">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>How can I help you?</h2>
              <p>
                Ask a question about the documents in your knowledge base.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={`${index}-${message.role}`}
              className={`message-row ${message.role}`}
            >
              <div className="message-content">
                <span className="message-author">
                  {message.role === "human" ? "You" : "AI Assistant"}
                </span>

                <div className="message-bubble markdown-content">
                  {message.role === "ai" ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    message.content
                  )}
                </div>

                {/* Source references */}
                {message.sources && message.sources.length > 0 && (
                  <details className="sources-panel">
                    <summary>
                      Sources ({message.sources.length})
                    </summary>

                    {message.sources.map((source, sourceIndex) => (
                      <div
                        className="source-item"
                        key={source.chunk_id}
                      >
                        <strong>
                          {sourceIndex + 1}. {source.document_name}
                        </strong>

                        <p>
                          Type: {source.file_type.toUpperCase()}
                          {source.page_number !== null &&
                            ` · Page ${source.page_number}`}
                        </p>

                        <p>
                          Version: {source.document_version}
                          {" · "}
                          Distance: {source.distance.toFixed(4)}
                        </p>

                        <p className="source-content">
                          {source.content}
                        </p>
                      </div>
                    ))}
                  </details>
                )}
              </div>
            </div>
          ))}

          {sending && (
            <div className="typing-indicator">
              AI Assistant is thinking...
            </div>
          )}

          <div ref={messagesEndRef} />
        </section>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* Message input */}
        <footer className="chat-input-area">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something about your documents..."
            rows={2}
            disabled={!activeSessionId || sending}
          />

          <button
            className="send-button"
            onClick={handleSendMessage}
            disabled={!question.trim() || !activeSessionId || sending}
          >
            {sending ? "Sending..." : "Send"}
          </button>

          <p>Press Enter to send · Shift + Enter for a new line</p>
        </footer>
      </main>
    </div>
  );
}

export default App;
