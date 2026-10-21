
import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const LABELS = {
  limitations: "Limitations",
  missing_experiments: "Missing experiments",
  weak_evaluation: "Evaluation concerns",
  future_opportunities: "Future research",
};

export default function ResearchGaps({ jobId }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!jobId || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_BASE}/phase2/research-gaps/${jobId}`,
        { method: "POST" }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Research gap analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      className="panel"
      aria-labelledby="research-gaps-heading"
    >
      <h2 id="research-gaps-heading">Research Gaps</h2>

      <p className="note">
        Find candidate limitations, experiment concerns, and
        future research directions from the uploaded paper.
      </p>

      <button
        className="primary"
        type="button"
        onClick={analyze}
        disabled={!jobId || loading}
      >
        {loading ? "Analyzing…" : "Analyze research gaps"}
      </button>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <div aria-live="polite">
          <h3>Analysis overview</h3>

          <p>
            Candidate findings: {result.total_candidates}
          </p>

          <div className="chips">
            {Object.entries(result.counts).map(([key, count]) => (
              <span className="chip" key={key}>
                {LABELS[key] || key}: {count}
              </span>
            ))}
          </div>

          <p className="note">{result.notice}</p>

          {result.research_gaps.length === 0 && (
            <p>
              No candidate passages matched the current rules.
              This does not mean the paper has no research gaps.
            </p>
          )}

          {result.research_gaps.map((gap) => (
            <article className="chunk" key={gap.id}>
              <h3>{gap.title}</h3>

              <p>
                <strong>Category:</strong>{" "}
                {LABELS[gap.category] || gap.category}
              </p>

              <p>
                <strong>Evidence from paper:</strong>
              </p>

              <blockquote>{gap.evidence}</blockquote>

              <p>{gap.interpretation}</p>

              <p>
                <strong>Possible next step:</strong>{" "}
                {gap.suggested_investigation}
              </p>

              <small>
                Evidence status: author-stated candidate.
                Verify the context before treating this as a confirmed gap.
              </small>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
