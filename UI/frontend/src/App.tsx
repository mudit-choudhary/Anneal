import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import Chat from "./components/Chat";
import Ingestion from "./components/Ingestion";
import SettingsPage from "./components/Settings";
import Sidebar from "./components/Sidebar";
import StatusBar from "./components/StatusBar";
import type { ChatSummary, Status } from "./types";

type View = "chat" | "ingestion" | "settings";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [status, setStatus] = useState<Status | null>(null);
  const [papers, setPapers] = useState<string[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);

  const refreshStatus = useCallback(() => api.status().then(setStatus).catch(() => setStatus(null)), []);
  const refreshPapers = useCallback(() => api.papers().then((r) => setPapers(r.papers)).catch(() => setPapers([])), []);
  const refreshChats = useCallback(() => api.chats().then((r) => setChats(r.chats)).catch(() => setChats([])), []);

  useEffect(() => {
    refreshStatus();
    refreshPapers();
    refreshChats();
    const t = setInterval(refreshStatus, 10000);
    return () => clearInterval(t);
  }, [refreshStatus, refreshPapers, refreshChats]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          Research Paper <span>RAG</span>
        </h1>
        <nav>
          {(["chat", "ingestion", "settings"] as View[]).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => setView(v)}>
              {v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        <StatusBar status={status} />
      </header>
      <main className="layout">
        {view === "chat" && (
          <>
            <Sidebar
              papers={papers}
              selected={selected}
              setSelected={setSelected}
              onRefreshPapers={refreshPapers}
              chats={chats}
              chatId={chatId}
              onSelectChat={setChatId}
              onNewChat={() => setChatId(null)}
              onChatsChanged={refreshChats}
            />
            <Chat
              chatId={chatId}
              filenames={selected.size ? [...selected] : null}
              backend={status?.backend ?? "local"}
              onChatStarted={(id) => {
                setChatId(id);
                refreshChats();
              }}
              onChatsChanged={refreshChats}
            />
          </>
        )}
        {view === "ingestion" && <Ingestion />}
        {view === "settings" && <SettingsPage onSaved={refreshStatus} />}
      </main>
    </div>
  );
}
