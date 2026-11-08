
const API_BASE_URL = "http://127.0.0.1:8000";

export interface Session {
  session_id: string;
  created_at: string;
}

export interface Message {
    role: "system" | "human" | "ai";
    content: string;
    sources?: Source[];
  }

export interface SessionDetails {
  session_id: string;
  messages: Message[];
}

export interface Source {
  chunk_id: string;
  document_id: string;
  document_name: string;
  document_version: number;
  file_type: string;
  source: string;
  page_number: number | null;
  distance: number;
  content: string;
}

export interface ChatResponse {
  session_id: string;
  question: string;
  answer: string;
  sources: Source[];
  standalone_question: string | null;
  validation: {
    status: string;
    supported_claims: string[];
    unsupported_claims: string[];
    corrected_answer: string;
  } | null;
  needs_clarification: boolean;
}

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Something went wrong");
  }

  return data as T;
}

export const api = {
  getSessions: () =>
    request<Session[]>("/sessions"),

  createSession: () =>
    request<{ session_id: string }>("/sessions", {
      method: "POST",
    }),

  getSession: (sessionId: string) =>
    request<SessionDetails>(
      `/sessions/${encodeURIComponent(sessionId)}`
    ),

  sendMessage: (sessionId: string, question: string) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        question,
      }),
    }),

  deleteSession: (sessionId: string) =>
    request<{ message: string }>(
      `/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    ),
};
