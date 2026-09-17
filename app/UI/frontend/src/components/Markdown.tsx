import { useEffect, useId, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";

/** Diagrams must follow the app theme, or they render dark-on-light. */
function mermaidTheme(): "default" | "dark" {
  return document.documentElement.dataset.theme === "light" ? "default" : "dark";
}
// suppressErrorRendering: otherwise every failed render leaves Mermaid's
// "Syntax error" bomb graphic appended to <body>.
const config = () => ({ startOnLoad: false, theme: mermaidTheme(), securityLevel: "strict" as const,
                        suppressErrorRendering: true });
mermaid.initialize(config());

/** Models write `A[Query Rewriting (Optional)]`; brackets and similar inside an
 *  unquoted label break the parser. Quote such labels. */
export function quoteLabels(chart: string) {
  return chart.replace(/(\b[\w-]+)\[([^\]"\n]*[(){}<>|;:#&][^\]"\n]*)\]/g,
                       (m, node, label: string) => /^\(.*\)$/.test(label) ? m   // [(db)] is a shape
                         : `${node}["${label.replace(/"/g, "'")}"]`);
}

/** Re-renders diagrams when the theme attribute on <html> changes. */
function useThemeName() {
  const [name, setName] = useState(() => document.documentElement.dataset.theme ?? "dark");
  useEffect(() => {
    const obs = new MutationObserver(() => setName(document.documentElement.dataset.theme ?? "dark"));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return name;
}

function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, "");
  const [svg, setSvg] = useState<string>("");
  const [err, setErr] = useState<string>("");
  const theme = useThemeName();
  useEffect(() => {
    let cancelled = false;
    mermaid.initialize(config());
    setErr("");
    const fixed = quoteLabels(chart);
    mermaid
      .render(`m${id}`, chart)
      .catch((e) => (fixed !== chart ? mermaid.render(`m${id}`, fixed) : Promise.reject(e)))
      .then(({ svg }) => !cancelled && setSvg(svg))
      .catch((e) => !cancelled && setErr(String(e?.message ?? e).split("\n")[0]));
    return () => {
      cancelled = true;
    };
  }, [chart, id, theme]);
  if (err) {
    return (
      <div className="mermaid-error">
        <div className="note">⚠ This diagram has a syntax error, so here is its source. ({err})</div>
        <pre><code>{chart}</code></pre>
      </div>
    );
  }
  return <ZoomableDiagram svg={svg} />;
}

/** Diagrams are drawn to fit the bubble, which makes wide ones unreadable.
 *  Click opens the same SVG in a native <dialog> (Esc closes) where it can be
 *  scaled up and scrolled. */
function ZoomableDiagram({ svg }: { svg: string }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [zoom, setZoom] = useState(1);

  return (
    <>
      <div className="mermaid" title="Click to enlarge"
           onClick={() => { setZoom(1); dialog.current?.showModal(); }}
           dangerouslySetInnerHTML={{ __html: svg }} />
      <dialog ref={dialog} className="diagram-zoom" onClose={() => setZoom(1)}>
        <div className="bar">
          <button className="small" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}>−</button>
          <span>{Math.round(zoom * 100)}%</span>
          <button className="small" onClick={() => setZoom((z) => Math.min(6, z + 0.25))}>+</button>
          <button className="small" onClick={() => setZoom(1)}>Reset</button>
          <button className="small" onClick={() => dialog.current?.close()}>Close ✕</button>
        </div>
        <div className="pane">
          <div className="sizer" style={{ width: `${zoom * 100}%` }}
               dangerouslySetInnerHTML={{ __html: svg }} />
        </div>
      </dialog>
    </>
  );
}

/** Markdown with GFM tables, ```mermaid diagrams, and [n] citations as chips. */
export default function Markdown({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  // turn [3] into a link the renderer can style as a chip
  const withCites = text.replace(/\[(\d{1,2})\]/g, (_m, n) => `[[${n}]](#cite-${n})`);
  // Stable component identities: a fresh object here would make React treat
  // every code block as a new type and remount it — diagrams would be redrawn
  // (and an open zoom dialog destroyed) on each re-render of the chat.
  const cite = useRef(onCite);
  cite.current = onCite;
  const components = useMemo(() => ({
    a({ href, children, ...rest }: any) {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.slice(6));
        return (
          <a className="cite" href={href} onClick={(e) => { e.preventDefault(); cite.current?.(n); }}>
            {children}
          </a>
        );
      }
      return <a href={href} target="_blank" rel="noreferrer" {...rest}>{children}</a>;
    },
    code({ className, children, ...rest }: any) {
      const lang = /language-(\w+)/.exec(className || "")?.[1];
      const body = String(children).replace(/\n$/, "");
      if (lang === "mermaid") return <Mermaid chart={body} />;
      return <code className={className} {...rest}>{children}</code>;
    },
  }), []);
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {withCites}
    </ReactMarkdown>
  );
}
