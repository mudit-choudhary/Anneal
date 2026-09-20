import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import Chat from "./components/Chat";
import ChatList from "./components/ChatList";
import Ingestion from "./components/Ingestion";
import PaperFilter from "./components/PaperFilter";
import SettingsPage from "./components/Settings";
import StatusBar from "./components/StatusBar";
import type { ChatSummary, Paper, Status } from "./types";
import Gpu from "./components/Gpu";
import Logs from "./components/Logs";
import ThemeToggle from "./components/ThemeToggle";

type View = "chat" | "ingestion" | "gpu" | "logs" | "settings";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [status, setStatus] = useState<Status | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
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
          Ann<span>eal</span>
        </h1>
        <nav>
          {(["chat", "ingestion", "gpu", "logs", "settings"] as View[]).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => setView(v)}>
              {v === "gpu" ? "GPU" : v[0].toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        <StatusBar status={status} />
        {/* AGPL-3.0 section 13: a networked app must offer its source to users. */}
        <a className="source" href="https://github.com/mudit-choudhary/Anneal" target="_blank" rel="noreferrer"
           title="Anneal is free software (AGPL-3.0) — source code">source</a>
        <ThemeToggle />
      </header>
      <main className="layout">
        {view === "chat" && (
          <>
            <ChatList
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
              model={(status?.backend === "openai" ? status?.openai_model : status?.model) ?? ""}
              paperCount={papers.length}
              onChatStarted={(id) => {
                setChatId(id);
                refreshChats();
              }}
              onChatsChanged={refreshChats}
            />
            <PaperFilter
              papers={papers}
              selected={selected}
              setSelected={setSelected}
              onRefresh={refreshPapers}
            />
          </>
        )}
        {view === "ingestion" && <Ingestion />}
        {view === "gpu" && <Gpu />}
        {view === "logs" && <Logs />}
        {view === "settings" && <SettingsPage onSaved={refreshStatus} />}
      </main>
    </div>
  );
}
