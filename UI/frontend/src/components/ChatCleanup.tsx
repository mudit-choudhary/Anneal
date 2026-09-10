import { useEffect, useState } from "react";
import { api } from "../api";
import type { ChatSummary } from "../types";

/** Settings: clear saved conversations in bulk.
 *  Two scopes, because they carry very different risk. "No answer" removes only
 *  the debris of interrupted or failed questions and is always safe; "everything"
 *  destroys real conversation history and asks for the count to be typed back. */
export default function ChatCleanup() {
  const [chats, setChats] = useState<ChatSummary[] | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmText, setConfirmText] = useState("");

  const reload = () =>
    api.chats().then((r) => setChats(r.chats)).catch((e) => setMsg(String(e.message)));

  useEffect(() => { reload(); }, []);

  // The API decides what counts as unanswered; this is only for the label, so
  // an odd message count (a question with no reply) is a good enough estimate.
  const total = chats?.length ?? 0;
  const unanswered = (chats ?? []).filter((c) => c.n_messages % 2 === 1).length;
  const inMemory = (chats ?? []).filter((c) => c.embedded).length;

  async function run(scope: "all" | "unanswered") {
    setBusy(true);
    setMsg("");
    try {
      const r = await api.deleteChats({ scope });
      setMsg(r.note ?? `Deleted ${r.deleted} chat${r.deleted === 1 ? "" : "s"}.`);
      setConfirmText("");
      await reload();
    } catch (e) {
      setMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="danger">
      <h3>Saved conversations</h3>
      <p className="hint">
        {total} saved{inMemory > 0 && `, ${inMemory} also embedded into searchable memory`}.
        Deleting a chat removes it from memory too, so it stops turning up as a source.
      </p>

      <div className="row">
        <button className="small" disabled={busy || unanswered === 0}
                onClick={() => run("unanswered")}>
          Delete {unanswered} with no answer
        </button>
        <span className="hint">
          Questions that were interrupted or failed before the model replied.
        </span>
      </div>

      <div className="row">
        <input style={{ width: 90 }} value={confirmText} placeholder={`type ${total}`}
               disabled={busy || total === 0}
               onChange={(e) => setConfirmText(e.target.value)} />
        <button className="small warn"
                disabled={busy || total === 0 || confirmText.trim() !== String(total)}
                onClick={() => run("all")}>
          Delete all {total} chats
        </button>
        <span className="hint">
          Type the number to confirm. This cannot be undone.
        </span>
      </div>

      {msg && <p className="hint">{msg}</p>}
    </div>
  );
}
