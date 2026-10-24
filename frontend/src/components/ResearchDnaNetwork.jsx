import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchJson } from "../config";

const VIEW_WIDTH = 840;
const VIEW_HEIGHT = 560;
const IDENTITY = { x: 0, y: 0, scale: 1 };
const MIN_SCALE = 0.5;
const MAX_SCALE = 3;

const REL_COLORS = {
  uses: "#56744a",
  improves: "#cf3a38",
  evaluated_by: "#3d6f94",
  depends_on: "#8a6a2f",
  compared_with: "#7a4f8f",
  produces: "#2f8a7e",
  co_occurs_with: "#87909a",
};
const relColor = (type) => REL_COLORS[type] || "#7d8a74";
const relLabel = (edge) =>
  edge.relationship_label || String(edge.relationship_type || "related").replaceAll("_", " ");

/* ---------------------------------------------------------------------- */
/* Deterministic force-directed layout: same graph -> same picture.       */
/* ---------------------------------------------------------------------- */
function computeLayout(rawNodes, rawEdges) {
  const cx = VIEW_WIDTH / 2;
  const cy = VIEW_HEIGHT / 2;
  const n = rawNodes.length;

  const nodes = rawNodes.map((node, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    const r = Math.max(26, Math.min(52, 22 + Math.sqrt(Number(node.percentage) || 0) * 4));
    return {
      ...node,
      label: String(node.label || "Untitled topic").trim(),
      r,
      x: cx + Math.cos(angle) * 170,
      y: cy + Math.sin(angle) * 120,
      vx: 0,
      vy: 0,
    };
  });

  const index = Object.fromEntries(nodes.map((node, i) => [node.id, i]));
  const links = rawEdges
    .map((e) => [index[e.source], index[e.target]])
    .filter(([a, b]) => a !== undefined && b !== undefined && a !== b);

  const ITER = 360;
  for (let step = 0; step < ITER; step++) {
    const alpha = 1 - step / ITER;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) {
          dx = (i - j) * 0.5 || 0.5;
          dy = 0.5;
          d2 = 1;
        }
        const d = Math.sqrt(d2);
        const push = (9000 * alpha) / d2;
        const minDist = a.r + b.r + 30;
        const overlap = d < minDist ? (minDist - d) * 0.5 : 0;
        const f = push + overlap / d;
        a.vx -= (dx / d) * f;
        a.vy -= (dy / d) * f;
        b.vx += (dx / d) * f;
        b.vy += (dy / d) * f;
      }
    }

    for (const [ai, bi] of links) {
      const a = nodes[ai];
      const b = nodes[bi];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const rest = a.r + b.r + 90;
      const f = (d - rest) * 0.025 * alpha;
      a.vx += (dx / d) * f;
      a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f;
      b.vy -= (dy / d) * f;
    }

    for (const node of nodes) {
      node.vx += (cx - node.x) * 0.012 * alpha;
      node.vy += (cy - node.y) * 0.012 * alpha;
      node.vx *= 0.82;
      node.vy *= 0.82;
      node.x = Math.max(node.r + 12, Math.min(VIEW_WIDTH - node.r - 12, node.x + node.vx));
      node.y = Math.max(node.r + 12, Math.min(VIEW_HEIGHT - node.r - 12, node.y + node.vy));
    }
  }
  return nodes;
}

function wrapLabel(label, maxChars = 16) {
  const words = String(label || "").split(/\s+/).filter(Boolean);
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
  if (lines.length > 2) {
    lines.length = 2;
    lines[1] = `${lines[1].slice(0, maxChars - 1)}…`;
  }
  return lines;
}

function EvidenceList({ evidence }) {
  if (!evidence?.length) return <p className="note">No supporting passage is available.</p>;
  return (
    <div className="dna-evidence">
      {evidence.map((item, index) => (
        <blockquote key={`${item.page ?? "p"}-${index}`}>
          <p>{item.text || "Evidence text was not supplied."}</p>
          <small>
            {item.status === "validated_relation_cue"
              ? "Relation cue"
              : item.status === "co_occurrence"
              ? "Co-occurrence evidence"
              : "Source passage"}
            {item.cue ? ` “${item.cue}”` : ""}
            {item.page != null ? ` · Page ${item.page}` : ""}
            {item.section ? ` · ${item.section}` : ""}
          </small>
        </blockquote>
      ))}
    </div>
  );
}

export default function ResearchDnaNetwork({ jobId }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [selected, setSelected] = useState(null); // {kind, id}
  const [topicsOpen, setTopicsOpen] = useState(false);
  const [hovered, setHovered] = useState(null);
  const [transform, setTransform] = useState(IDENTITY);
  const drag = useRef(null);
  const canvasRef = useRef(null);
  const svgRef = useRef(null);

  useEffect(() => {
    setSelected(null);
    setHovered(null);
    setTransform(IDENTITY);
    setError("");
    setGraph(null);
    if (!jobId) {
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    fetchJson(`/phase2/topic-network/${encodeURIComponent(jobId)}`, { signal: controller.signal })
      .then(setGraph)
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message || "Could not load topic network.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [jobId, reloadKey]);

  const layout = useMemo(() => {
    if (!graph?.nodes?.length) return { nodes: [], edges: [], byId: {} };
    const nodes = computeLayout(graph.nodes, graph.edges || []);
    const byId = Object.fromEntries(nodes.map((nd) => [nd.id, nd]));

    // Parallel edges between the same pair get fanned out so they stay clickable.
    const pairCount = {};
    const pairSeen = {};
    for (const e of graph.edges || []) {
      const key = [e.source, e.target].sort().join("|");
      pairCount[key] = (pairCount[key] || 0) + 1;
    }
    const edges = (graph.edges || [])
      .map((e) => {
        const a = byId[e.source];
        const b = byId[e.target];
        if (!a || !b) return null;
        const key = [e.source, e.target].sort().join("|");
        const k = (pairSeen[key] = (pairSeen[key] ?? -1) + 1);
        const offset = (k - (pairCount[key] - 1) / 2) * 14;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const ux = dx / d;
        const uy = dy / d;
        const px = -uy * offset;
        const py = ux * offset;
        const directed = e.directed !== false;
        return {
          ...e,
          a,
          b,
          directed,
          x1: a.x + ux * a.r + px,
          y1: a.y + uy * a.r + py,
          x2: b.x - ux * (b.r + (directed ? 5 : 0)) + px,
          y2: b.y - uy * (b.r + (directed ? 5 : 0)) + py,
        };
      })
      .filter(Boolean);
    return { nodes, edges, byId };
  }, [graph]);

  const relationTypes = useMemo(
    () => [...new Set(layout.edges.map((e) => e.relationship_type))],
    [layout.edges]
  );

  const active = hovered || selected;
  const neighbourhood = useMemo(() => {
    if (!active) return null;
    if (active.kind === "edge") {
      const e = layout.edges.find((x) => x.id === active.id);
      return e ? { nodes: new Set([e.source, e.target]), edges: new Set([e.id]) } : null;
    }
    const nodes = new Set([active.id]);
    const edges = new Set();
    for (const e of layout.edges) {
      if (e.source === active.id || e.target === active.id) {
        nodes.add(e.source);
        nodes.add(e.target);
        edges.add(e.id);
      }
    }
    return { nodes, edges };
  }, [active, layout.edges]);

  /* ----- zoom / pan ------------------------------------------------------ */
  const zoomAt = useCallback((factor, px = VIEW_WIDTH / 2, py = VIEW_HEIGHT / 2) => {
    setTransform((t) => {
      const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, t.scale * factor));
      const f = scale / t.scale;
      return { scale, x: px - (px - t.x) * f, y: py - (py - t.y) * f };
    });
  }, []);

  // Ctrl/Cmd + wheel (also trackpad pinch) zooms; plain scroll still scrolls the page.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return undefined;
    const onWheel = (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomAt(
        event.deltaY < 0 ? 1.12 : 1 / 1.12,
        ((event.clientX - rect.left) * VIEW_WIDTH) / rect.width,
        ((event.clientY - rect.top) * VIEW_HEIGHT) / rect.height
      );
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt, layout.nodes.length]);

  const onPointerDown = (event) => {
    if (event.button !== 0) return;
    if (event.target.closest?.(".dna-node, .dna-edge")) return;
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
    const d = drag.current;
    if (!d || d.pointerId !== event.pointerId) return;
    const rect = svgRef.current.getBoundingClientRect();
    const k = VIEW_WIDTH / rect.width; // screen px -> SVG units
    const dx = event.clientX - d.startX;
    const dy = event.clientY - d.startY;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (d.moved) setTransform((t) => ({ ...t, x: d.originX + dx * k, y: d.originY + dy * k }));
  };
  const onPointerUp = (event) => {
    if (drag.current?.pointerId === event.pointerId) {
      if (!drag.current.moved) setSelected(null); // click on empty canvas clears selection
      drag.current = null;
    }
  };

  const activateOnKey = (select) => (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      select();
    }
  };
  const selectNode = (id) => setSelected({ kind: "node", id });
  const selectEdge = (id) => setSelected({ kind: "edge", id });

  /* ----- states ---------------------------------------------------------- */
  if (!jobId) return <p className="note">Upload and process a paper to explore its Research DNA.</p>;

  if (loading) {
    return (
      <section className="panel">
        <h2>Research DNA</h2>
        <p role="status">Building topic network from extracted text… the first run can take a little while.</p>
      </section>
    );
  }
  if (error) {
    return (
      <section className="panel">
        <h2>Research DNA</h2>
        <p className="error" role="alert">{error}</p>
        <button type="button" className="ghost" onClick={() => setReloadKey((k) => k + 1)}>
          Try again
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

  const selectedItem =
    selected?.kind === "node"
      ? layout.byId[selected.id]
      : selected?.kind === "edge"
      ? layout.edges.find((e) => e.id === selected.id)
      : null;

  const connections =
    selected?.kind === "node"
      ? layout.edges.filter((e) => e.source === selected.id || e.target === selected.id)
      : [];

  return (
    <section className="panel dna-panel" aria-labelledby="dna-title">
      <div className="row dna-header">
        <div>
          <h2 id="dna-title">Research DNA</h2>
          <p className="note tight">
            Circle size reflects each topic’s share of assigned topic mass. Arrows show explicit relationships; grey lines show topics mentioned together in a source sentence.
          </p>
        </div>
        <div className="dna-zoom" role="group" aria-label="Graph controls">
          <button type="button" className="ghost" onClick={() => zoomAt(1.2)} aria-label="Zoom in">+</button>
          <button type="button" className="ghost" onClick={() => zoomAt(1 / 1.2)} aria-label="Zoom out">−</button>
          <button type="button" className="ghost" onClick={() => setTransform(IDENTITY)}>Reset view</button>
        </div>
      </div>

      <div className="dna-layout">
        <div>
          <div
            ref={canvasRef}
            className="dna-canvas"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
          >
            <svg
              ref={svgRef}
              viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
              role="group"
              aria-label="Interactive Research DNA topic network"
            >
              <defs>
                {Object.keys(REL_COLORS).concat("other").map((type) => (
                  <marker
                    key={type}
                    id={`arrow-${type}`}
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="8"
                    markerHeight="8"
                    orient="auto"
                    markerUnits="userSpaceOnUse"
                  >
                    <path d="M0 0 L10 5 L0 10 z" fill={relColor(type)} />
                  </marker>
                ))}
              </defs>
              <rect className="dna-background" width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="transparent" />
              <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
                {layout.edges.map((edge) => {
                  const dim = neighbourhood && !neighbourhood.edges.has(edge.id);
                  const isSel = selected?.kind === "edge" && selected.id === edge.id;
                  const color = relColor(edge.relationship_type);
                  const showLabel = isSel || (neighbourhood?.edges.has(edge.id) && active?.kind === "edge");
                  const title = `${edge.a.label} ${relLabel(edge)} ${edge.b.label}`;
                  return (
                    <g
                      key={edge.id}
                      className={`dna-edge ${isSel ? "selected" : ""} ${dim ? "dim" : ""}`}
                      role="button"
                      tabIndex={0}
                      aria-label={title}
                      onClick={(e) => { e.stopPropagation(); selectEdge(edge.id); }}
                      onKeyDown={activateOnKey(() => selectEdge(edge.id))}
                      onMouseEnter={() => setHovered({ kind: "edge", id: edge.id })}
                      onMouseLeave={() => setHovered(null)}
                      onFocus={() => setHovered({ kind: "edge", id: edge.id })}
                      onBlur={() => setHovered(null)}
                    >
                      <line x1={edge.x1} y1={edge.y1} x2={edge.x2} y2={edge.y2}
                        stroke="transparent" strokeWidth="18" pointerEvents="stroke" />
                      <line
                        className="dna-edge-line"
                        x1={edge.x1}
                        y1={edge.y1}
                        x2={edge.x2}
                        y2={edge.y2}
                        stroke={color}
                        strokeWidth={isSel ? 3.5 : 2.5}
                        strokeOpacity={dim ? 0.12 : 0.95}
                        strokeLinecap="round"
                        markerEnd={
                          edge.directed
                            ? `url(#arrow-${
                                REL_COLORS[edge.relationship_type]
                                  ? edge.relationship_type
                                  : "other"
                              })`
                            : undefined
                        }
                        pointerEvents="none"
                      />
                      {showLabel && (
                        <text className="dna-edge-label" x={(edge.x1 + edge.x2) / 2} y={(edge.y1 + edge.y2) / 2 - 6}
                          textAnchor="middle" pointerEvents="none">
                          {relLabel(edge)}
                        </text>
                      )}
                      <title>{title}</title>
                    </g>
                  );
                })}

                {layout.nodes.map((node) => {
                  const dim = neighbourhood && !neighbourhood.nodes.has(node.id);
                  const isSel = selected?.kind === "node" && selected.id === node.id;
                  const lines = wrapLabel(node.label);
                  const pct = `${Number(node.percentage).toFixed(1)}%`;
                  return (
                    <g
                      key={node.id}
                      className={`dna-node ${isSel ? "selected" : ""} ${dim ? "dim" : ""}`}
                      transform={`translate(${node.x} ${node.y})`}
                      role="button"
                      tabIndex={0}
                      aria-label={`${node.label}, ${Number(node.percentage).toFixed(2)} percent`}
                      onClick={(e) => { e.stopPropagation(); selectNode(node.id); }}
                      onKeyDown={activateOnKey(() => selectNode(node.id))}
                      onMouseEnter={() => setHovered({ kind: "node", id: node.id })}
                      onMouseLeave={() => setHovered(null)}
                      onFocus={() => setHovered({ kind: "node", id: node.id })}
                      onBlur={() => setHovered(null)}
                    >
                      <circle r={node.r} />
                      <text className="dna-label" textAnchor="middle" pointerEvents="none"
                        y={-(lines.length - 1) * 6 - 2}>
                        {lines.map((line, i) => (
                          <tspan key={i} x="0" dy={i === 0 ? 0 : 12}>{line}</tspan>
                        ))}
                      </text>
                      <text className="dna-percent" textAnchor="middle" pointerEvents="none"
                        y={(lines.length - 1) * 6 + 14}>
                        {pct}
                      </text>
                      <title>{`${node.label}: ${Number(node.percentage).toFixed(2)}%`}</title>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>

          {relationTypes.length > 0 && (
            <ul className="dna-legend" aria-label="Relationship legend">
              {relationTypes.map((type) => (
                <li key={type}>
                  <span className="dna-swatch" style={{ background: relColor(type) }} />
                  {layout.edges.find((e) => e.relationship_type === type) && relLabel(layout.edges.find((e) => e.relationship_type === type))}
                </li>
              ))}
            </ul>
          )}
          {layout.edges.length === 0 && (
            <p className="note">
              No topic relationships were stated explicitly in the text, so no arrows are drawn.
            </p>
          )}
          <p className="note">
            Click a topic or arrow to inspect it. Drag empty space to pan; Ctrl/⌘ + scroll to zoom.
          </p>
        </div>

        
<aside className="dna-topics" aria-label="Topics ranked by contribution">
  <button
    type="button"
    className="dna-topics-toggle"
    onClick={() => setTopicsOpen((open) => !open)}
    aria-expanded={topicsOpen}
  >
    <span>
      Topics ({layout.nodes.length})
    </span>
    <span className={`dna-topics-chevron ${topicsOpen ? "open" : ""}`}>
      ▾
    </span>
  </button>

  {topicsOpen && (
    <ol>
      {[...layout.nodes]
        .sort((a, b) => b.percentage - a.percentage)
        .map((node) => (
          <li key={node.id}>
            <button
              type="button"
              className={`dna-topic-btn ${
                selected?.kind === "node" && selected.id === node.id
                  ? "selected"
                  : ""
              }`}
              onClick={() => selectNode(node.id)}
              onMouseEnter={() => setHovered({ kind: "node", id: node.id })}
              onMouseLeave={() => setHovered(null)}
            >
              <span className="dna-topic-name">{node.label}</span>
              <span className="dna-topic-pct">
                {Number(node.percentage).toFixed(1)}%
              </span>
              <span className="dna-bar">
                <span
                  style={{
                    width: `${Math.min(100, node.percentage)}%`,
                  }}
                />
              </span>
            </button>
          </li>
        ))}
    </ol>
  )}
</aside>

      </div>

      {selectedItem && (
        <div className="dna-detail" aria-live="polite">
          <div className="row dna-detail-header">
            <h3>
              {selected.kind === "node"
                ? selectedItem.label
                : `${selectedItem.a.label} ${selectedItem.directed ? "→" : "↔"} ${selectedItem.b.label}`}
            </h3>
            <button type="button" className="ghost" onClick={() => setSelected(null)}>Close</button>
          </div>

          {selected.kind === "node" ? (
            <>
              <p>
                <strong>{Number(selectedItem.percentage).toFixed(2)}%</strong> of the paper’s topic coverage ·{" "}
                {selectedItem.passage_support} supporting passage{selectedItem.passage_support === 1 ? "" : "s"}
                {selectedItem.mentions != null && <> · named literally in {selectedItem.mentions} sentence{selectedItem.mentions === 1 ? "" : "s"}</>}
              </p>
              {selectedItem.sections?.length > 0 && (
                <p><strong>Mostly discussed in:</strong> {selectedItem.sections.join(", ")}</p>
              )}
              {connections.length > 0 && (
                <>
                  <p><strong>Connections</strong></p>
                  <ul className="dna-connections">
                    {connections.map((e) => (
                      <li key={e.id}>
                        <button type="button" className="linklike" onClick={() => selectEdge(e.id)}>
                          {e.a.label} {relLabel(e)} {e.b.label}
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              <EvidenceList evidence={selectedItem.evidence} />
            </>
          ) : (
            <>
              <p>
                <strong>{selectedItem.a.label}</strong> {relLabel(selectedItem)} <strong>{selectedItem.b.label}</strong>
                {selectedItem.directed ? "" : " (symmetric relation)"}
              </p>
              <p>
                Supported by {selectedItem.evidence?.length || 0} source sentence
                {(selectedItem.evidence?.length || 0) === 1 ? "" : "s"}. Line thickness reflects how many
                distinct sentences state it.
              </p>
              <EvidenceList evidence={selectedItem.evidence} />
            </>
          )}
        </div>
      )}


    </section>
  );
}
