import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchJson } from "../config";
import "./ResearchGaps.css";

const CATEGORIES = {
  limitations: "Limitations",
  missing_experiments: "Missing experiments",
  weak_evaluation: "Evaluation concerns",
  future_opportunities: "Future research",
};
const CONFIDENCE_ORDER = { high: 0, medium: 1, low: 2 };
const PAGE_SIZE = 12;

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Wrap matched phrases in <mark>. Plain React nodes, no innerHTML. */
function Highlighted({ text, phrases }) {
  const parts = useMemo(() => {
    const list = (phrases || []).filter(Boolean).sort((a, b) => b.length - a.length);
    if (!list.length) return [{ text, hit: false }];
    const rx = new RegExp(`(${list.map(escapeRegExp).join("|")})`, "gi");
    return text
      .split(rx)
      .filter((part) => part !== "")
      .map((part) => ({ text: part, hit: list.some((p) => p.toLowerCase() === part.toLowerCase()) }));
  }, [text, phrases]);

  return parts.map((part, i) =>
    part.hit ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>
  );
}

function toMarkdown(result, gaps) {
  const lines = [`# Research gap candidates`, "", `> ${result.notice}`, ""];
  for (const [key, label] of Object.entries(CATEGORIES)) {
    const group = gaps.filter((g) => g.category === key);
    if (!group.length) continue;
    lines.push(`## ${label}`, "");
    for (const g of group) {
      const where = [g.section, g.page != null ? `p. ${g.page}` : null].filter(Boolean).join(", ");
      lines.push(`- **[${g.confidence}]** ${g.evidence}${where ? ` _(${where})_` : ""}`);
      lines.push(`  - Next step: ${g.suggested_investigation}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function GapCard({ gap }) {
  const [copied, setCopied] = useState(false);
  const where = [gap.section, gap.page != null ? `Page ${gap.page}` : null].filter(Boolean).join(" · ");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(gap.evidence);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <article className={`gap-card conf-${gap.confidence}`}>
      <header className="gap-card-head">
        <span className={`badge badge-cat cat-${gap.category}`}>{CATEGORIES[gap.category] || gap.category}</span>
        <span className={`badge badge-conf conf-${gap.confidence}`} title={`Score ${gap.score}`}>
          {gap.confidence} confidence
        </span>
        {gap.scope === "prior_work" && (
          <span className="badge badge-scope" title="The sentence criticises other people's work, not this paper">
            about prior work
          </span>
        )}
        {where && <span className="gap-where">{where}</span>}
      </header>

      <blockquote className="gap-quote">
        <Highlighted text={gap.evidence} phrases={gap.matched_phrases} />
      </blockquote>

      <p className="gap-interp">{gap.interpretation}</p>
      <p className="gap-next">
        <strong>Possible next step:</strong> {gap.suggested_investigation}
      </p>

      {(gap.context_before || gap.context_after) && (
        <details className="gap-context">
          <summary>Show surrounding text</summary>
          {gap.context_before && <p className="ctx">{gap.context_before}</p>}
          <p className="ctx ctx-main">
            <Highlighted text={gap.evidence} phrases={gap.matched_phrases} />
          </p>
          {gap.context_after && <p className="ctx">{gap.context_after}</p>}
        </details>
      )}

      <button type="button" className="ghost gap-copy" onClick={copy}>
        {copied ? "Copied" : "Copy quote"}
      </button>
    </article>
  );
}

export default function ResearchGaps({ jobId }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [category, setCategory] = useState("all");
  const [minConfidence, setMinConfidence] = useState("low");
  const [includePrior, setIncludePrior] = useState(false);
  const [query, setQuery] = useState("");
  const [visible, setVisible] = useState(PAGE_SIZE);

  // New paper -> forget the old analysis, and restore a saved one if it exists.
  useEffect(() => {
    setResult(null);
    setError("");
    setCategory("all");
    setQuery("");
    setVisible(PAGE_SIZE);
    if (!jobId) return undefined;

    const controller = new AbortController();
    fetchJson(`/phase2/research-gaps/${encodeURIComponent(jobId)}`, { signal: controller.signal })
      .then(setResult)
      .catch(() => {}); // 404 just means "not analysed yet"
    return () => controller.abort();
  }, [jobId]);

  const analyze = useCallback(async () => {
    if (!jobId || loading) return;
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson(`/phase2/research-gaps/${encodeURIComponent(jobId)}`, { method: "POST" });
      setResult(data);
      setVisible(PAGE_SIZE);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [jobId, loading]);

  const filtered = useMemo(() => {
    if (!result) return [];
    const q = query.trim().toLowerCase();
    const limit = CONFIDENCE_ORDER[minConfidence];
    return result.research_gaps.filter(
      (g) =>
        (category === "all" || g.category === category) &&
        CONFIDENCE_ORDER[g.confidence] <= limit &&
        (includePrior || g.scope !== "prior_work") &&
        (!q || g.evidence.toLowerCase().includes(q) || (g.section || "").toLowerCase().includes(q))
    );
  }, [result, category, minConfidence, includePrior, query]);

  const categoryCounts = useMemo(() => {
    const counts = {};
    (result?.research_gaps || []).forEach((g) => {
      if (includePrior || g.scope !== "prior_work") counts[g.category] = (counts[g.category] || 0) + 1;
    });
    return counts;
  }, [result, includePrior]);

  const shownTotal = Object.values(categoryCounts).reduce((a, b) => a + b, 0);

  return (
    <section className="panel gaps-panel" aria-labelledby="research-gaps-heading">
      <div className="row gaps-header">
        <div>
          <h2 id="research-gaps-heading">Research Gaps</h2>
          <p className="note tight">
            Candidate limitations, evaluation concerns and future directions, ranked by how clearly
            the authors state them.
          </p>
        </div>
        <button className="primary" type="button" onClick={analyze} disabled={!jobId || loading}>
          {loading ? "Analyzing…" : result ? "Re-run analysis" : "Analyze research gaps"}
        </button>
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {loading && !result && (
        <div className="gap-skeleton" aria-hidden="true">
          <span /> <span /> <span />
        </div>
      )}

      {result && (
        <div aria-live="polite">
          <p className="note gaps-summary">
            Scanned {result.sentences_scanned ?? "?"} sentences
            {result.input_mode === "plain_text" && " (plain text: no page or section info)"}.{" "}
            {result.confidence_counts && (
              <>
                {result.confidence_counts.high} high · {result.confidence_counts.medium} medium ·{" "}
                {result.confidence_counts.low} low confidence.
              </>
            )}
          </p>

          <div className="gap-tabs" role="tablist" aria-label="Filter by category">
            <button
              type="button"
              role="tab"
              aria-selected={category === "all"}
              className={`gap-tab ${category === "all" ? "active" : ""}`}
              onClick={() => { setCategory("all"); setVisible(PAGE_SIZE); }}
            >
              All <span>{shownTotal}</span>
            </button>
            {Object.entries(CATEGORIES).map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={category === key}
                className={`gap-tab ${category === key ? "active" : ""}`}
                onClick={() => { setCategory(key); setVisible(PAGE_SIZE); }}
                disabled={!categoryCounts[key]}
              >
                {label} <span>{categoryCounts[key] || 0}</span>
              </button>
            ))}
          </div>

          <div className="gap-controls">
            <input
              type="search"
              className="gap-search"
              placeholder="Search findings or sections…"
              aria-label="Search findings"
              value={query}
              onChange={(e) => { setQuery(e.target.value); setVisible(PAGE_SIZE); }}
            />
            <label className="gap-select">
              Minimum confidence
              <select value={minConfidence} onChange={(e) => { setMinConfidence(e.target.value); setVisible(PAGE_SIZE); }}>
                <option value="low">Show all</option>
                <option value="medium">Medium and high</option>
                <option value="high">High only</option>
              </select>
            </label>
            <label className="gap-check">
              <input type="checkbox" checked={includePrior} onChange={(e) => { setIncludePrior(e.target.checked); setVisible(PAGE_SIZE); }} />
              Include criticism of prior work ({result.scope_counts?.prior_work ?? 0})
            </label>
          </div>

          <p className="note">{result.notice}</p>

          {result.research_gaps.length === 0 && (
            <p>No candidate passages matched. This does not mean the paper has no research gaps.</p>
          )}
          {result.research_gaps.length > 0 && filtered.length === 0 && (
            <p>No findings match these filters.</p>
          )}

          <div className="gap-list">
            {filtered.slice(0, visible).map((gap) => (
              <GapCard key={gap.id} gap={gap} />
            ))}
          </div>

          {filtered.length > visible && (
            <button type="button" className="ghost gap-more" onClick={() => setVisible((v) => v + PAGE_SIZE)}>
              Show {Math.min(PAGE_SIZE, filtered.length - visible)} more ({filtered.length - visible} remaining)
            </button>
          )}

          {filtered.length > 0 && (
            <div className="gap-export">
              <button type="button" className="ghost" onClick={() => download(`research-gaps-${jobId}.md`, toMarkdown(result, filtered), "text/markdown")}>
                Export as Markdown
              </button>
              <button type="button" className="ghost" onClick={() => download(`research-gaps-${jobId}.json`, JSON.stringify({ ...result, research_gaps: filtered }, null, 2), "application/json")}>
                Export as JSON
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
