import { useRef, useState } from "react";
import ResearchDnaNetwork from "./components/ResearchDnaNetwork";
import "./components/ResearchDnaNetwork.css";

const API_BASE = "http://127.0.0.1:8000";
const STEPS = ["Upload", "Read the paper", "Summarise"];

async function api(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);

  let data = null;

  try {
    data = await res.json();
  } catch {
    /* non-JSON error body */
  }

  if (!res.ok) {
    const d = data?.detail;

    throw new Error(
      typeof d === "string"
        ? d
        : d
          ? JSON.stringify(d)
          : `Request failed (${res.status}).`
    );
  }

  return data;
}

const Sprig = () => (
  <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
    <path
      d="M16 29V8"
      stroke="#56744A"
      strokeWidth="2"
      strokeLinecap="round"
      fill="none"
    />
    <path
      d="M16 22c-5 0-8-3-8-7 5 0 8 3 8 7zM16 17c5 0 8-3 8-7-5 0-8 3-8 7z"
      fill="#56744A"
    />
    <circle cx="16" cy="6" r="3.5" fill="#CF3A38" />
  </svg>
);

export default function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [stage, setStage] = useState("idle");
  const [error, setError] = useState(null);
  const [parsed, setParsed] = useState(null);
  const [p2, setP2] = useState(null);
  const [quality, setQuality] = useState("fast");
  const [dragOver, setDragOver] = useState(false);
  const [copied, setCopied] = useState(false);

  const [query, setQuery] = useState("");
  const [hits, setHits] = useState(null);
  const [searching, setSearching] = useState(false);

  // ------------------------------------------------------------
  // Research interests / relevance
  // ------------------------------------------------------------

  const [domain, setDomain] = useState("");
  const [subs, setSubs] = useState([]);
  const [subInput, setSubInput] = useState("");
  const [rel, setRel] = useState(null);
  const [relLoading, setRelLoading] = useState(false);

  const busy =
    stage === "parsing" ||
    stage === "summarising";

  const activeStep =
    stage === "parsing"
      ? 1
      : stage === "summarising"
        ? 2
        : stage === "done"
          ? 3
          : 0;

  // ------------------------------------------------------------
  // File selection
  // ------------------------------------------------------------

  const pick = (f) => {
    if (!f) return;

    if (
      f.type !== "application/pdf" &&
      !f.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("That file isn't a PDF. Choose a .pdf file.");
      return;
    }

    setFile(f);
    setError(null);
    setParsed(null);
    setP2(null);
    setHits(null);
    setRel(null);
    setStage("idle");
  };

  // ------------------------------------------------------------
  // Upload + Phase 2 processing
  // ------------------------------------------------------------

  const run = async () => {
    if (!file || busy) return;

    setError(null);
    setParsed(null);
    setP2(null);
    setHits(null);
    setRel(null);
    setStage("parsing");

    try {
      const form = new FormData();
      form.append("file", file);

      const up = await api("/upload", {
        method: "POST",
        body: form,
      });

      setParsed(up);

      setStage("summarising");

      const out = await api(
        `/phase2/process/${up.job_id}?quality=${quality}`,
        {
          method: "POST",
        }
      );

      setP2(out);
      setStage("done");

      // If the user has already entered a domain,
      // automatically calculate relevance after processing.
      if (domain.trim().length >= 2) {
        await checkRelevance(up.job_id);
      }
    } catch (e) {
      setError(
        e.message === "Failed to fetch"
          ? "Can't reach the backend at 127.0.0.1:8000. Check that uvicorn is running."
          : e.message
      );

      setStage("idle");
    }
  };

  // ------------------------------------------------------------
  // Subdomain handling
  // ------------------------------------------------------------

  const addSub = (raw) => {
    const parts = raw
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);

    if (!parts.length) return subs;

    const next = [...subs];

    parts.forEach((p) => {
      if (
        next.length < 8 &&
        !next.some(
          (x) => x.toLowerCase() === p.toLowerCase()
        )
      ) {
        next.push(p);
      }
    });

    setSubs(next);
    setSubInput("");

    return next;
  };

  // ------------------------------------------------------------
  // Relevance API
  // ------------------------------------------------------------

  const checkRelevance = async (
    jobId = parsed?.job_id
  ) => {
    if (
      !jobId ||
      domain.trim().length < 2
    ) {
      return;
    }

    setRelLoading(true);
    setError(null);

    try {
      const list = subInput.trim()
        ? addSub(subInput)
        : subs;

      const result = await api(
        `/phase2/relevance/${jobId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            domain: domain.trim(),
            subdomains: list,
          }),
        }
      );

      setRel(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setRelLoading(false);
    }
  };

  // ------------------------------------------------------------
  // Relevance score styling
  // ------------------------------------------------------------

  const tone = (n) => {
    if (n >= 70) return "good";
    if (n >= 45) return "mid";
    return "low";
  };

  // ------------------------------------------------------------
  // Semantic search
  // ------------------------------------------------------------

  const search = async (e) => {
    e.preventDefault();

    if (!query.trim() || !parsed) return;

    setSearching(true);
    setError(null);

    try {
      const r = await api(
        `/phase2/search/${parsed.job_id}?q=${encodeURIComponent(
          query
        )}&k=5`
      );

      setHits(r.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  };

  // ------------------------------------------------------------
  // Copy summary
  // ------------------------------------------------------------

  const copy = async () => {
    await navigator.clipboard.writeText(
      p2.summary
    );

    setCopied(true);

    setTimeout(
      () => setCopied(false),
      1500
    );
  };

  return (
    <>
      <div
        className="ribbon"
        aria-hidden="true"
      />

      <div
        className="scallop"
        aria-hidden="true"
      />

      <main className="app">

        {/* -------------------------------------------------- */}
        {/* Header */}
        {/* -------------------------------------------------- */}

        <header>
          <div className="brand">
            <Sprig />
            <h1>ResearchScope</h1>
          </div>

          <p className="tagline">
            Add a research paper to get a short
            summary, key terms, and a way to
            search inside it.
          </p>
        </header>

        {/* -------------------------------------------------- */}
        {/* Upload */}
        {/* -------------------------------------------------- */}

        <section className="panel">

          <label
            className={`drop ${
              dragOver ? "over" : ""
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() =>
              setDragOver(false)
            }
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              pick(
                e.dataTransfer.files[0]
              );
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              onChange={(e) =>
                pick(e.target.files[0])
              }
            />

            <Sprig />

            <span className="drop-title">
              {file
                ? file.name
                : "Drop a PDF here, or click to choose one"}
            </span>

            <span className="drop-hint">
              {file
                ? `${(
                    file.size / 1048576
                  ).toFixed(1)} MB`
                : "Text-based PDFs up to 50 MB"}
            </span>
          </label>

          <div className="controls">

            <div
              className="seg"
              role="radiogroup"
              aria-label="Summary quality"
            >
              {[
                ["fast", "Fast"],
                ["best", "Better quality"],
              ].map(([v, l]) => (
                <button
                  key={v}
                  type="button"
                  role="radio"
                  aria-checked={
                    quality === v
                  }
                  className={
                    quality === v
                      ? "on"
                      : ""
                  }
                  disabled={busy}
                  onClick={() =>
                    setQuality(v)
                  }
                >
                  {l}
                </button>
              ))}
            </div>

            <button
              className="primary"
              onClick={run}
              disabled={!file || busy}
            >
              {busy
                ? "Working…"
                : "Summarise paper"}
            </button>

          </div>

          <ol
            className="steps"
            aria-label="Progress"
          >
            {STEPS.map((s, i) => (
              <li
                key={s}
                className={
                  i < activeStep
                    ? "done"
                    : i === activeStep &&
                        busy
                      ? "active"
                      : ""
                }
              >
                <span className="dot">
                  {i < activeStep
                    ? "✓"
                    : i === activeStep &&
                        busy
                      ? "✦"
                      : ""}
                </span>

                {s}
              </li>
            ))}
          </ol>

          {stage === "summarising" && (
            <p className="note">
              Summarising can take a few
              minutes, especially the first time
              while the models load.
            </p>
          )}

        </section>

        {/* -------------------------------------------------- */}
        {/* Research interests */}
        {/* -------------------------------------------------- */}

        <section
          className="panel"
          aria-labelledby="in-h"
        >

          <h2 id="in-h">
            Your research interests
          </h2>

          <p className="note tight">
            Tell us what you study. We'll score
            how well the paper fits.
          </p>

          <div className="field">

            <label htmlFor="domain">
              Domain
            </label>

            <input
              id="domain"
              value={domain}
              maxLength={100}
              onChange={(e) =>
                setDomain(e.target.value)
              }
              placeholder="e.g. Machine learning"
            />

          </div>

          <div className="field">

            <label htmlFor="subs">
              Subdomains{" "}
              <span className="opt">
                (optional, press Enter to add)
              </span>
            </label>

            <div className="tags">

              {subs.map((t) => (
                <span
                  className="tag"
                  key={t}
                >
                  {t}

                  <button
                    type="button"
                    aria-label={`Remove ${t}`}
                    onClick={() =>
                      setSubs(
                        subs.filter(
                          (x) => x !== t
                        )
                      )
                    }
                  >
                    ×
                  </button>
                </span>
              ))}

              <input
                id="subs"
                value={subInput}
                disabled={
                  subs.length >= 8
                }
                onChange={(e) =>
                  setSubInput(
                    e.target.value
                  )
                }
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" ||
                    e.key === ","
                  ) {
                    e.preventDefault();
                    addSub(subInput);
                  } else if (
                    e.key ===
                      "Backspace" &&
                    !subInput &&
                    subs.length
                  ) {
                    setSubs(
                      subs.slice(0, -1)
                    );
                  }
                }}
                placeholder={
                  subs.length
                    ? ""
                    : "e.g. demand forecasting"
                }
              />

            </div>
          </div>

          <div className="controls left">

            <button
              className="primary"
              onClick={() =>
                checkRelevance()
              }
              disabled={
                !p2 ||
                domain.trim().length < 2 ||
                relLoading
              }
            >
              {relLoading
                ? "Checking…"
                : "Check relevance"}
            </button>

            {!p2 && (
              <span className="note inline">
                Available once the paper is
                summarised.
              </span>
            )}

          </div>

          {/* ------------------------------------------------ */}
          {/* Relevance result */}
          {/* ------------------------------------------------ */}

          {rel && (
            <div
              className="relevance"
              aria-live="polite"
            >

              <div className="score-row">

                <div
                  className={`score ${tone(
                    rel.score
                  )}`}
                >
                  {rel.score}
                  <small>/100</small>
                </div>

                <div className="score-main">

                  <strong>
                    {rel.label}
                  </strong>

                  <div className="bar">
                    <i
                      className={tone(
                        rel.score
                      )}
                      style={{
                        width: `${rel.score}%`,
                      }}
                    />
                  </div>

                </div>

              </div>

              <p className="why">
                {rel.justification}
              </p>

              <ul className="terms">

                {[
                  rel.domain,
                  ...rel.subdomains,
                ].map((t) => (
                  <li key={t.term}>

                    <div className="term-head">

                      <span>
                        {t.term}
                      </span>

                      <span className="term-score">
                        {t.score}
                      </span>

                    </div>

                    <div className="bar sm">
                      <i
                        className={tone(
                          t.score
                        )}
                        style={{
                          width: `${t.score}%`,
                        }}
                      />
                    </div>

                    {t.score >= 25 && (
                      <p className="evidence">
                        Page{" "}
                        {t.evidence.page}:{" "}
                        {t.evidence.text}
                      </p>
                    )}

                  </li>
                ))}

              </ul>

            </div>
          )}

        </section>

        {/* -------------------------------------------------- */}
        {/* Error */}
        {/* -------------------------------------------------- */}

        {error && (
          <div
            className="error"
            role="alert"
          >
            {error}
          </div>
        )}

        {/* -------------------------------------------------- */}
        {/* Phase 2 results */}
        {/* -------------------------------------------------- */}

        {p2 && (
          <>

            {/* Research DNA topic network */}
            <ResearchDnaNetwork jobId={parsed?.job_id} />

            {/* Summary */}

            <section
              className="panel"
              aria-labelledby="sum-h"
            >

              <div className="row">

                <h2 id="sum-h">
                  Summary
                </h2>

                <button
                  className="ghost"
                  onClick={copy}
                >
                  {copied
                    ? "Copied"
                    : "Copy summary"}
                </button>

              </div>

              <p className="summary">
                {p2.summary}
              </p>

            </section>

            {/* Keywords */}

            <section
              className="panel"
              aria-labelledby="kw-h"
            >

              <h2 id="kw-h">
                Key terms
              </h2>

              <div className="chips">

                {p2.keywords.map((k) => (
                  <span
                    className="chip"
                    key={k.keyword}
                  >
                    {k.keyword}
                  </span>
                ))}

              </div>

              {p2.keywords_document?.length >
                0 && (
                <>
                  <h3>
                    Across the whole paper
                  </h3>

                  <div className="chips alt">

                    {p2.keywords_document.map(
                      (k) => (
                        <span
                          className="chip"
                          key={k.keyword}
                        >
                          {k.keyword}
                        </span>
                      )
                    )}

                  </div>
                </>
              )}

            </section>

            {/* Search */}

            <section
              className="panel"
              aria-labelledby="se-h"
            >

              <h2 id="se-h">
                Search this paper
              </h2>

              <form
                className="search"
                onSubmit={search}
              >

                <input
                  value={query}
                  onChange={(e) =>
                    setQuery(
                      e.target.value
                    )
                  }
                  placeholder='Ask about a topic, e.g. "how is demand predicted?"'
                  aria-label="Search query"
                />

                <button
                  className="primary"
                  disabled={
                    searching ||
                    query.trim().length <
                      2
                  }
                >
                  {searching
                    ? "Searching…"
                    : "Search"}
                </button>

              </form>

              {hits &&
                hits.length === 0 && (
                  <p className="note">
                    No matching passages
                    found.
                  </p>
                )}

              <ul className="hits">

                {hits?.map((h) => (
                  <li key={h.id}>

                    <p>
                      {h.text}
                    </p>

                    <div className="meta">

                      <span>
                        Page {h.page} ·{" "}
                        {h.section}
                      </span>

                      <span
                        className="match"
                        title="Similarity"
                      >

                        <i
                          style={{
                            width: `${Math.max(
                              4,
                              Math.round(
                                h.score * 100
                              )
                            )}%`,
                          }}
                        />

                        {Math.round(
                          h.score * 100
                        )}
                        % match

                      </span>

                    </div>

                  </li>
                ))}

              </ul>

            </section>

            {/* Details */}

            <section
              className="panel"
              aria-labelledby="de-h"
            >

              <h2 id="de-h">
                Details
              </h2>

              <dl className="stats">

                <div>
                  <dt>Pages</dt>
                  <dd>
                    {parsed.num_pages}
                  </dd>
                </div>

                <div>
                  <dt>Sentences</dt>
                  <dd>
                    {p2.stats.sentences}
                  </dd>
                </div>

                <div>
                  <dt>
                    Repeats removed
                  </dt>
                  <dd>
                    {
                      p2.stats
                        .removed_redundant
                    }
                  </dd>
                </div>

                <div>
                  <dt>
                    Sections summarised
                  </dt>
                  <dd>
                    {p2.stats.chunks}
                  </dd>
                </div>

              </dl>

              <div className="links">

                <a
                  href={`${API_BASE}${parsed.json_download}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download parsed JSON
                </a>

                <a
                  href={`${API_BASE}${parsed.txt_download}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download parsed text
                </a>

              </div>

              <details>

                <summary>
                  Summary of each part (
                  {p2.chunks.length}
                  )
                </summary>

                {p2.chunks.map((c) => (
                  <article
                    key={c.chunk_id}
                    className="chunk"
                  >

                    <h4>
                      {c.sections.join(", ")}
                      <small>
                        {" "}
                        (page{" "}
                        {c.pages.join(", ")})
                      </small>
                    </h4>

                    <p>
                      {c.summary}
                    </p>

                  </article>
                ))}

              </details>

            </section>

          </>
        )}

      </main>
    </>
  );
}