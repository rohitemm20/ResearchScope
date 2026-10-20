import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function ResearchDnaNetwork({ jobId }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const drag = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setLoading(true); setError(""); setGraph(null); setSelected(null);
    fetch(`${API_BASE}/phase2/topic-network/${encodeURIComponent(jobId)}`)
      .then(async r => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
        return data;
      })
      .then(data => { if (!cancelled) setGraph(data); })
      .catch(e => { if (!cancelled) setError(e.message || "Could not load topic network."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [jobId]);

  const layout = useMemo(() => {
    if (!graph?.nodes?.length) return { nodes: [], edges: [] };
    const cx = 420, cy = 280, rx = 310, ry = 205;
    const nodes = graph.nodes.map((n, i) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * i / graph.nodes.length;
      return { ...n, x: cx + rx * Math.cos(angle), y: cy + ry * Math.sin(angle),
        r: Math.max(24, Math.min(48, 20 + Math.sqrt(n.percentage || 0) * 3.4)) };
    });
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
    return { nodes, edges: (graph.edges || []).map(e => ({ ...e, a: byId[e.source], b: byId[e.target] })).filter(e => e.a && e.b) };
  }, [graph]);

  const zoom = factor => setTransform(t => ({ ...t, scale: Math.max(.5, Math.min(2.5, t.scale * factor)) }));
  const onPointerDown = e => { drag.current = { x: e.clientX - transform.x, y: e.clientY - transform.y }; e.currentTarget.setPointerCapture(e.pointerId); };
  const onPointerMove = e => { if (drag.current) setTransform(t => ({ ...t, x: e.clientX - drag.current.x, y: e.clientY - drag.current.y })); };
  const onPointerUp = () => { drag.current = null; };

  if (!jobId) return <p className="note">Upload and process a paper to explore its Research DNA.</p>;
  if (loading) return <section className="panel"><h2>Research DNA</h2><p role="status">Building topic network from extracted text…</p></section>;
  if (error) return <section className="panel"><h2>Research DNA</h2><p className="error" role="alert">{error}</p><button className="ghost" onClick={() => setGraph(null) || setError("")}>Dismiss</button></section>;
  if (!graph?.nodes?.length) return <section className="panel"><h2>Research DNA</h2><p className="note">{graph?.message || "No supported topics were found in this paper."}</p></section>;

  const selectEdge = edge => setSelected({ kind: "edge", item: edge });
  const selectNode = node => setSelected({ kind: "node", item: node });

  return <section className="panel dna-panel" aria-labelledby="dna-title">
    <div className="row"><div><h2 id="dna-title">Research DNA</h2><p className="note tight">Topics are sized by relative semantic passage contribution. Edges require explicit textual evidence.</p></div>
      <div className="dna-zoom"><button className="ghost" onClick={() => zoom(1.15)} aria-label="Zoom in">+</button><button className="ghost" onClick={() => zoom(1/1.15)} aria-label="Zoom out">−</button><button className="ghost" onClick={() => setTransform({x:0,y:0,scale:1})}>Reset view</button></div>
    </div>
    <div className="dna-canvas" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
      <svg viewBox="0 0 840 560" role="img" aria-label="Interactive network of paper topics and evidence-supported relationships">
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
          {layout.edges.map(edge => <g key={edge.id} className="dna-edge" onClick={e => {e.stopPropagation(); selectEdge(edge);}} role="button" tabIndex="0" aria-label={`${edge.relationship_type} relationship`} onKeyDown={e => e.key === "Enter" && selectEdge(edge)}>
            <line x1={edge.a.x} y1={edge.a.y} x2={edge.b.x} y2={edge.b.y} strokeWidth={1.5 + edge.strength * 2} />
            <title>{edge.relationship_type}</title>
          </g>)}
          {layout.nodes.map(node => <g key={node.id} className={`dna-node ${selected?.item?.id === node.id ? "selected" : ""}`} transform={`translate(${node.x} ${node.y})`} onClick={e => {e.stopPropagation(); selectNode(node);}} role="button" tabIndex="0" aria-label={`${node.label}, ${node.percentage}%`} onKeyDown={e => e.key === "Enter" && selectNode(node)}>
            <circle r={node.r} /><text textAnchor="middle" y="-3">{node.label.length > 22 ? node.label.slice(0, 20) + "…" : node.label}</text><text className="dna-percent" textAnchor="middle" y="15">{Number(node.percentage).toFixed(2)}%</text>
            <title>{node.label}: {node.percentage}%</title>
          </g>)}
        </g>
      </svg>
    </div>
    <p className="note">Tip: drag to pan; use +/− to zoom. Select a node or an edge to inspect details.</p>
    {selected && <div className="dna-detail" aria-live="polite">
      <div className="row"><h3>{selected.kind === "node" ? selected.item.label : `${selected.item.relationship_type.replaceAll("_", " ")} relationship`}</h3><button className="ghost" onClick={() => setSelected(null)}>Close</button></div>
      {selected.kind === "node" ? <>
        <p><strong>Contribution:</strong> {Number(selected.item.percentage).toFixed(2)}% of retained topic mass</p>
        <p>{selected.item.description}</p><p><strong>Supporting passages:</strong> {selected.item.passage_support}</p>
        <EvidenceList evidence={selected.item.evidence || []} />
      </> : <>
        <p><strong>Topics:</strong> {layout.nodes.find(n => n.id === selected.item.source)?.label} ↔ {layout.nodes.find(n => n.id === selected.item.target)?.label}</p>
        <p><strong>Type:</strong> {selected.item.relationship_type.replaceAll("_", " ")}</p>
        <p>{selected.item.explanation}</p><p><strong>Edge strength:</strong> {selected.item.strength} (evidence sentence support; not a topic percentage)</p>
        <p><strong>Evidence status:</strong> {selected.item.evidence_status}</p><EvidenceList evidence={selected.item.evidence || []} />
      </>}
    </div>}
    <details className="dna-method"><summary>How percentages and relationships are calculated</summary><p>{graph.method?.contribution}</p><p>{graph.method?.relationship_validation}</p><p>Percentages describe relative semantic passage coverage among retained topics, not word frequency, query similarity, or confidence. Values may change if extraction/model versions change.</p></details>
  </section>;
}

function EvidenceList({ evidence }) {
  if (!evidence.length) return <p className="note">No supporting passage is available.</p>;
  return <div className="dna-evidence">{evidence.map((ev, i) => <blockquote key={`${ev.page}-${i}-${ev.text.slice(0,20)}`}>
    <p>{ev.text}</p><small>Evidence status: {ev.status}{ev.page ? ` · Page ${ev.page}` : ""}{ev.section ? ` · ${ev.section}` : ""}{ev.block_ids?.length ? ` · Blocks ${ev.block_ids.join(", ")}` : ""}</small>
  </blockquote>)}</div>;
}
