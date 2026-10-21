import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";
const VIEW_WIDTH = 840;
const VIEW_HEIGHT = 560;

export default function ResearchDnaNetwork({ jobId }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const drag = useRef(null);

  useEffect(() => {
    if (!jobId) {
      setGraph(null);
      setError("");
      setLoading(false);
      setSelected(null);
      return undefined;
    }

    const controller = new AbortController();
    setLoading(true);
    setError("");
    setGraph(null);
    setSelected(null);
    setTransform({ x: 0, y: 0, scale: 1 });

    fetch(`${API_BASE}/phase2/topic-network/${encodeURIComponent(jobId)}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        return data;
      })
      .then((data) => setGraph(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(err.message || "Could not load topic network.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [jobId]);

  const layout = useMemo(() => {
    if (!graph?.nodes?.length) return { nodes: [], edges: [] };

    const cx = VIEW_WIDTH / 2;
    const cy = VIEW_HEIGHT / 2;
    const rx = Math.min(310, VIEW_WIDTH * 0.39);
    const ry = Math.min(205, VIEW_HEIGHT * 0.37);
    const count = graph.nodes.length;

    const nodes = graph.nodes.map((node, index) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / count;
      const label = String(node.label || "Untitled topic").trim();
      return {
        ...node,
        label,
        x: cx + rx * Math.cos(angle),
        y: cy + ry * Math.sin(angle),
        r: Math.max(27, Math.min(48, 22 + Math.sqrt(Number(node.percentage) || 0) * 3)),
      };
    });

    const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
    const edges = (graph.edges || [])
      .map((edge) => ({ ...edge, a: byId[edge.source], b: byId[edge.target] }))
      .filter((edge) => edge.a && edge.b);

    return { nodes, edges };
  }, [graph]);

  const zoom = (factor) => {
    setTransform((current) => ({
      ...current,
      scale: Math.max(0.6, Math.min(2.4, current.scale * factor)),
    }));
  };

  // Pan only when the user starts on empty canvas. Starting on a node/edge
  // never captures the pointer, so SVG click handlers can fire normally.
  const onPointerDown = (event) => {
    if (event.button !== 0) return;
    if (event.target.closest?.(".dna-node, .dna-edge, button")) return;

    drag.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: transform.x,
      originY: transform.y,
      moved: false,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event) => {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.current.startX;
    const dy = event.clientY - drag.current.startY;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
    if (drag.current.moved) {
      setTransform((current) => ({
        ...current,
        x: drag.current.originX + dx,
        y: drag.current.originY + dy,
      }));
    }
  };

  const onPointerUp = (event) => {
    if (drag.current?.pointerId === event.pointerId) drag.current = null;
  };

  const selectNode = (node) => setSelected({ kind: "node", item: node });
  const selectEdge = (edge) => setSelected({ kind: "edge", item: edge });

  if (!jobId) {
    return <p className="note">Upload and process a paper to explore its Research DNA.</p>;
  }
  if (loading) {
    return (
      <section className="panel">
        <h2>Research DNA</h2>
        <p role="status">Building topic network from extracted text…</p>
      </section>
    );
  }
  if (error) {
    return (
      <section className="panel">
        <h2>Research DNA</h2>
        <p className="error" role="alert">{error}</p>
        <button className="ghost" onClick={() => { setError(""); setGraph(null); }}>
          Dismiss
        </button>
      </section>
    );
  }
  if (!graph?.nodes?.length) {
    return (
      <section className="panel">
        <h2>Research DNA</h2>
        <p className="note">{graph?.message || "No supported topics were found in this paper."}</p>
      </section>
    );
  }

  const selectedSource = selected?.kind === "edge"
    ? layout.nodes.find((node) => node.id === selected.item.source)?.label
    : null;
  const selectedTarget = selected?.kind === "edge"
    ? layout.nodes.find((node) => node.id === selected.item.target)?.label
    : null;

  return (
    <section className="panel dna-panel" aria-labelledby="dna-title">
      <div className="row dna-header">
        <div>
          <h2 id="dna-title">Research DNA</h2>
          <p className="note tight">
            Node size shows relative topic contribution. Edges appear only when source text supports a relationship.
          </p>
        </div>
        <div className="dna-zoom" aria-label="Graph controls">
          <button type="button" className="ghost" onClick={() => zoom(1.15)} aria-label="Zoom in">+</button>
          <button type="button" className="ghost" onClick={() => zoom(1 / 1.15)} aria-label="Zoom out">−</button>
          <button type="button" className="ghost" onClick={() => setTransform({ x: 0, y: 0, scale: 1 })}>Reset view</button>
        </div>
      </div>

      <div
        className="dna-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <svg
          viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
          role="group"
          aria-label="Interactive Research DNA topic network"
        >
          <rect className="dna-background" width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="transparent" />
          <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
            {layout.edges.map((edge) => (
              <g
                key={edge.id}
                className={`dna-edge ${selected?.kind === "edge" && selected.item.id === edge.id ? "selected" : ""}`}
                role="button"
                tabIndex={0}
                aria-label={`${edge.a.label} — ${edge.relationship_type} — ${edge.b.label}`}
                onClick={(event) => { event.stopPropagation(); selectEdge(edge); }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    event.stopPropagation();
                    selectEdge(edge);
                  }
                }}
              >
                {/* Wide transparent stroke gives thin edges a usable click target. */}
                <line
                  className="dna-edge-hit"
                  x1={edge.a.x} y1={edge.a.y} x2={edge.b.x} y2={edge.b.y}
                  strokeWidth="18"
                  pointerEvents="stroke"
                />
                <line
                  className="dna-edge-line"
                  x1={edge.a.x} y1={edge.a.y} x2={edge.b.x} y2={edge.b.y}
                  strokeWidth={1.5 + (Number(edge.strength) || 0) * 2}
                  pointerEvents="none"
                />
                <title>{`${edge.a.label} — ${edge.relationship_type} — ${edge.b.label}`}</title>
              </g>
            ))}

            {layout.nodes.map((node) => (
              <g
                key={node.id}
                className={`dna-node ${selected?.kind === "node" && selected.item.id === node.id ? "selected" : ""}`}
                transform={`translate(${node.x} ${node.y})`}
                role="button"
                tabIndex={0}
                aria-label={`${node.label}, ${Number(node.percentage).toFixed(2)} percent`}
                onClick={(event) => { event.stopPropagation(); selectNode(node); }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    event.stopPropagation();
                    selectNode(node);
                  }
                }}
              >
                <circle r={node.r} />
                <text className="dna-label" textAnchor="middle" y="-4" pointerEvents="none">
                  {wrapLabel(node.label, 18).map((line, index) => (
                    <tspan key={`${node.id}-line-${index}`} x="0" dy={index === 0 ? 0 : 12}>
                      {line}
                    </tspan>
                  ))}
                </text>
                <text className="dna-percent" textAnchor="middle" y={Math.min(25, 8 + wrapLabel(node.label, 18).length * 11)} pointerEvents="none">
                  {`${Number(node.percentage).toFixed(2)}%`}
                </text>
                <title>{`${node.label}: ${Number(node.percentage).toFixed(2)}%`}</title>
              </g>
            ))}
          </g>
        </svg>
      </div>

      <p className="note">Select a node or edge to inspect it. Drag empty space to pan; use the controls to zoom.</p>

      {selected && (
        <div className="dna-detail" aria-live="polite">
          <div className="row dna-detail-header">
            <h3>{selected.kind === "node" ? selected.item.label : "Topic relationship"}</h3>
            <button type="button" className="ghost" onClick={() => setSelected(null)}>Close</button>
          </div>

          {selected.kind === "node" ? (
            <>
              <p><strong>Contribution:</strong> {Number(selected.item.percentage).toFixed(2)}% of retained topic mass</p>
              <p>{selected.item.description || "A topic identified from semantically supported source passages."}</p>
              <p><strong>Supporting passages:</strong> {selected.item.passage_support ?? (selected.item.evidence || []).length}</p>
              <EvidenceList evidence={selected.item.evidence || []} />
            </>
          ) : (
            <>
              <p><strong>Topics:</strong> {selectedSource || selected.item.source} ↔ {selectedTarget || selected.item.target}</p>
              <p><strong>Relationship type:</strong> {String(selected.item.relationship_type || "unspecified").replaceAll("_", " ")}</p>
              <p>{selected.item.explanation || "See the source evidence below; no additional explanation was supplied."}</p>
              <p><strong>Edge strength:</strong> {selected.item.strength ?? "Not supplied"} (separate from topic contribution)</p>
              <p><strong>Evidence status:</strong> {selected.item.evidence_status || "Not supplied"}</p>
              <EvidenceList evidence={selected.item.evidence || []} />
            </>
          )}
        </div>
      )}

      <details className="dna-method">
        <summary>How percentages and relationships are calculated</summary>
        <p>{graph.method?.contribution || "Topic percentages are normalized over the retained topic set."}</p>
        <p>{graph.method?.relationship_validation || "Relationships require source evidence; semantic similarity alone is insufficient."}</p>
        <p>Percentages represent relative semantic passage contribution, not word frequency, query similarity, or model confidence.</p>
      </details>
    </section>
  );
}

function wrapLabel(label, maxChars = 18) {
  const words = String(label || "Untitled topic").split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  // The full, unabridged label remains available in the SVG title and detail panel.
  if (lines.length > 2) {
    lines.length = 2;
    const last = lines[1];
    lines[1] = last.length >= maxChars ? `${last.slice(0, maxChars - 1)}…` : `${last}…`;
  }
  return lines;
}

function EvidenceList({ evidence }) {
  if (!evidence.length) return <p className="note">No supporting passage is available.</p>;
  return (
    <div className="dna-evidence">
      {evidence.map((item, index) => (
        <blockquote key={`${item.page ?? "page"}-${index}-${String(item.text || "").slice(0, 20)}`}>
          <p>{item.text || "Evidence text was not supplied."}</p>
          <small>
            Evidence status: {item.status || "unspecified"}
            {item.page != null ? ` · Page ${item.page}` : ""}
            {item.section ? ` · ${item.section}` : ""}
            {item.block_ids?.length ? ` · Blocks ${item.block_ids.join(", ")}` : ""}
          </small>
        </blockquote>
      ))}
    </div>
  );
}
