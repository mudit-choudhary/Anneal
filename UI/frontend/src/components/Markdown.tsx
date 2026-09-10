import { useEffect, useId, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import mermaid from "mermaid";

/** Diagrams must follow the app theme, or they render dark-on-light. */
function mermaidTheme() {
  return document.documentElement.dataset.theme === "light" ? "default" : "dark";
}
mermaid.initialize({ startOnLoad: false, theme: mermaidTheme(), securityLevel: "strict" });

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
    mermaid.initialize({ startOnLoad: false, theme: mermaidTheme(), securityLevel: "strict" });
    mermaid
      .render(`m${id}`, chart)
      .then(({ svg }) => !cancelled && setSvg(svg))
      .catch((e) => !cancelled && setErr(String(e?.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, [chart, id, theme]);
  if (err) return <pre className="mermaid-error">mermaid: {err}\n\n{chart}</pre>;
  return <div className="mermaid" dangerouslySetInnerHTML={{ __html: svg }} />;
}

/** Markdown with GFM tables, ```mermaid diagrams, and [n] citations as chips. */
export default function Markdown({ text, onCite }: { text: string; onCite?: (n: number) => void }) {
  // turn [3] into a link the renderer can style as a chip
  const withCites = text.replace(/\[(\d{1,2})\]/g, (_m, n) => `[[${n}]](#cite-${n})`);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a({ href, children, ...rest }) {
          if (href?.startsWith("#cite-")) {
            const n = Number(href.slice(6));
            return (
              <a className="cite" href={href} onClick={(e) => { e.preventDefault(); onCite?.(n); }}>
                {children}
              </a>
            );
          }
          return <a href={href} target="_blank" rel="noreferrer" {...rest}>{children}</a>;
        },
        code({ className, children, ...rest }) {
          const lang = /language-(\w+)/.exec(className || "")?.[1];
          const body = String(children).replace(/\n$/, "");
          if (lang === "mermaid") return <Mermaid chart={body} />;
          return <code className={className} {...rest}>{children}</code>;
        },
      }}
    >
      {withCites}
    </ReactMarkdown>
  );
}
