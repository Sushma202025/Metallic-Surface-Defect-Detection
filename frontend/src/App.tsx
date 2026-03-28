import { useState, useCallback, useRef, CSSProperties } from "react";

/* ─────────────────────────────────────────────────────────
   TYPES
───────────────────────────────────────────────────────── */
interface Prediction {
  defect: string;
  confidence: number;
  all_probs: Record<string, number>;
  gradcam_url?: string;
}

interface HistoryEntry {
  id: number;
  filename: string;
  defect: string;
  confidence: number;
  severity: string;
  timestamp: string;
  preview: string;
  all_probs: Record<string, number>;
}

type Tab = "detect" | "batch" | "history" | "stats";

/* ─────────────────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────────────────── */
const CLASSES = ["Crazing", "Inclusion", "Normal", "Patches", "Pitted", "Rolled", "Scratches"];

const DEFECT_META: Record<string, { desc: string; cause: string; action: string; severity: string; color: string; icon: string }> = {
  Crazing:   { desc: "Network of fine surface cracks under tensile stress.", cause: "Rapid cooling or mechanical stress", action: "Reject — structural risk", severity: "Critical", color: "#f43f5e", icon: "⚡" },
  Inclusion: { desc: "Foreign particles trapped during solidification.",      cause: "Contamination during casting",      action: "Inspect & reprocess",       severity: "High",     color: "#f97316", icon: "⬡" },
  Normal:    { desc: "No surface anomaly detected. Surface is acceptable.",  cause: "—",                                 action: "Approve for use",            severity: "None",     color: "#10b981", icon: "✓" },
  Patches:   { desc: "Irregular zones with material composition variation.", cause: "Uneven rolling / heat treatment",   action: "Detailed inspection",        severity: "Medium",   color: "#eab308", icon: "▣" },
  Pitted:    { desc: "Localised cavities formed by corrosion.",              cause: "Chemical exposure / humidity",      action: "Surface treatment required", severity: "High",     color: "#f97316", icon: "◉" },
  Rolled:    { desc: "Deformation streaks introduced during rolling.",       cause: "Rolling mill misalignment",         action: "Review rolling parameters",  severity: "Low",      color: "#60a5fa", icon: "≡" },
  Scratches: { desc: "Linear surface damage from abrasion or contact.",      cause: "Handling or transport damage",      action: "Light polishing may fix",    severity: "Low",      color: "#60a5fa", icon: "∕" },
};

const SEV_ORDER: Record<string, number> = { None: 0, Low: 1, Medium: 2, High: 3, Critical: 4 };

const CAROUSEL_ITEMS = [
  { title: "Crazing",    img: "https://images.unsplash.com/photo-1606337321936-02d1b1a4d5ef?w=700&auto=format&fit=crop" },
  { title: "Scratches",  img: "https://plus.unsplash.com/premium_photo-1769017353009-5dddd661ec2e?w=700&auto=format&fit=crop" },
  { title: "Inclusions", img: "https://images.unsplash.com/photo-1598302936625-6075fbd98dd7?w=700&auto=format&fit=crop" },
  { title: "Rolled",     img: "https://images.unsplash.com/photo-1509024368907-57294758cfc5?w=700&auto=format&fit=crop" },
  { title: "Pitted",     img: "https://images.unsplash.com/photo-1563733744821-2c00cc273851?w=700&auto=format&fit=crop" },
];

let _id = 0;

/* ─────────────────────────────────────────────────────────
   THEME
───────────────────────────────────────────────────────── */
const TK = {
  dark:  { bg: "#07090f", surface: "#0e1320", card: "#141b2d", border: "#1f2e4a", text: "#e8eaf0", muted: "#4a5a78", accent: "#38bdf8", navBg: "#07090f" },
  light: { bg: "#f0f4fa", surface: "#e4ecf7", card: "#ffffff",  border: "#c8d7ee", text: "#0d1526", muted: "#5a6f8a", accent: "#0284c7", navBg: "#ffffff" },
};

/* ─────────────────────────────────────────────────────────
   HELPERS
───────────────────────────────────────────────────────── */
function exportCSV(rows: HistoryEntry[]) {
  const header = "ID,Filename,Defect,Confidence,Severity,Timestamp";
  const body   = rows.map(r => `${r.id},"${r.filename}",${r.defect},${r.confidence}%,${r.severity},"${r.timestamp}"`);
  const blob   = new Blob([[header, ...body].join("\n")], { type: "text/csv" });
  const a      = document.createElement("a");
  a.href       = URL.createObjectURL(blob);
  a.download   = `metalinspect_${Date.now()}.csv`;
  a.click();
}

/* ─────────────────────────────────────────────────────────
   MAIN COMPONENT
───────────────────────────────────────────────────────── */
export default function App() {
  const [dark, setDark]           = useState(true);
  const [tab, setTab]             = useState<Tab>("detect");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [result, setResult]       = useState<Prediction | null>(null);
  const [loading, setLoading]     = useState(false);
  const [dragging, setDragging]   = useState(false);
  const [history, setHistory]     = useState<HistoryEntry[]>([]);
  const [carouselIdx, setCarouselIdx] = useState(0);
  const [zoomImg, setZoomImg]     = useState<string | null>(null);
  const [sortField, setSortField] = useState<keyof HistoryEntry>("id");
  const [sortAsc, setSortAsc]     = useState(false);
  const [filterDefect, setFilterDefect] = useState("All");

  /* batch */
  const [batchFiles, setBatchFiles]     = useState<File[]>([]);
  const [batchResults, setBatchResults] = useState<(HistoryEntry & { status: "pending"|"done"|"error" })[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);

  const fileInputRef  = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const t = dark ? TK.dark : TK.light;

  /* ── load file ── */
  const loadFile = (file: File) => {
    setImageFile(file); setPreview(URL.createObjectURL(file)); setResult(null);
  };
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.type.startsWith("image/")) loadFile(f);
  }, []);

  /* ── single predict ── */
  const analyse = async () => {
    if (!imageFile) return;
    const fd = new FormData(); fd.append("image", imageFile);
    setLoading(true); setResult(null);
    try {
      const res  = await fetch("http://127.0.0.1:5000/predict", { method: "POST", body: fd });
      const data: Prediction = await res.json();
      setResult(data);
      const entry: HistoryEntry = {
        id: ++_id, filename: imageFile.name,
        defect: data.defect, confidence: data.confidence,
        severity: DEFECT_META[data.defect]?.severity ?? "—",
        timestamp: new Date().toLocaleString(),
        preview: preview!, all_probs: data.all_probs ?? {},
      };
      setHistory(h => [entry, ...h].slice(0, 100));
    } catch { alert("⚠ Backend not reachable. Make sure app.py is running on port 5000."); }
    setLoading(false);
  };

  /* ── batch predict ── */
  const runBatch = async () => {
    if (!batchFiles.length) return;
    setBatchRunning(true);
    const initial = batchFiles.map(f => ({
      id: ++_id, filename: f.name, defect: "—", confidence: 0,
      severity: "—", timestamp: "", preview: URL.createObjectURL(f),
      all_probs: {}, status: "pending" as const,
    }));
    setBatchResults(initial);
    for (let i = 0; i < batchFiles.length; i++) {
      const fd = new FormData(); fd.append("image", batchFiles[i]);
      try {
        const res  = await fetch("http://127.0.0.1:5000/predict", { method: "POST", body: fd });
        const data: Prediction = await res.json();
        const ts = new Date().toLocaleString();
        const entry: HistoryEntry = {
          id: initial[i].id, filename: batchFiles[i].name,
          defect: data.defect, confidence: data.confidence,
          severity: DEFECT_META[data.defect]?.severity ?? "—",
          timestamp: ts, preview: initial[i].preview,
          all_probs: data.all_probs ?? {},
        };
        setBatchResults(prev => prev.map((r, idx) => idx === i ? { ...entry, status: "done" } : r));
        setHistory(h => [entry, ...h].slice(0, 100));
      } catch {
        setBatchResults(prev => prev.map((r, idx) => idx === i ? { ...r, status: "error" } : r));
      }
    }
    setBatchRunning(false);
  };

  /* ── history sort / filter ── */
  const sortedHistory = [...history]
    .filter(r => filterDefect === "All" || r.defect === filterDefect)
    .sort((a, b) => {
      let av: any = a[sortField], bv: any = b[sortField];
      if (sortField === "severity") { av = SEV_ORDER[av] ?? 0; bv = SEV_ORDER[bv] ?? 0; }
      return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });

  const toggleSort = (f: keyof HistoryEntry) => {
    if (sortField === f) setSortAsc(a => !a);
    else { setSortField(f); setSortAsc(false); }
  };

  /* ── stats ── */
  const total   = history.length;
  const defects = history.filter(r => r.defect !== "Normal").length;
  const passRate = total ? ((history.filter(r => r.defect === "Normal").length / total) * 100).toFixed(1) : "—";
  const avgConf  = total ? (history.reduce((s, r) => s + r.confidence, 0) / total).toFixed(1) : "—";
  const freqMap: Record<string, number> = {};
  history.forEach(r => { freqMap[r.defect] = (freqMap[r.defect] ?? 0) + 1; });
  const topDefect = Object.entries(freqMap).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";

  /* ══════════════════════════════════════════════
     RENDER
  ══════════════════════════════════════════════ */
  return (
    <div style={{ ...st.page, background: t.bg, color: t.text }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-thumb{background:#1f2e4a;border-radius:4px}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        .bar{transition:width .8s cubic-bezier(.22,1,.36,1)}
        .hist-row:hover{background:${t.surface} !important}
        .zoom-img:hover{transform:scale(1.015);cursor:zoom-in}
      `}</style>

      {/* ══ NAV ══ */}
      <nav style={{ ...st.nav, background: t.navBg, borderBottom: `1px solid ${t.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 30, height: 30, borderRadius: 7, background: t.accent, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Syne',sans-serif", fontWeight: 800, color: "#000", fontSize: 15 }}>M</span>
          <span style={{ fontFamily: "'Syne',sans-serif", fontSize: 16, fontWeight: 800, letterSpacing: 0.5 }}>MetalInspect</span>
          <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: t.surface, color: t.muted, letterSpacing: 1 }}>CNN+ViT</span>
        </div>
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          {(["detect", "batch", "history", "stats"] as Tab[]).map(tp => (
            <button key={tp} onClick={() => setTab(tp)} style={{
              fontSize: 10, padding: "6px 14px", borderRadius: 6, cursor: "pointer",
              letterSpacing: 1.5, fontFamily: "'Space Grotesk',sans-serif",
              color: tab === tp ? t.accent : t.muted,
              background: tab === tp ? t.card : "transparent",
              border: tab === tp ? `1px solid ${t.border}` : "1px solid transparent",
            }}>{tp.toUpperCase()}</button>
          ))}
          <button onClick={() => setDark(d => !d)} style={{
            width: 34, height: 34, borderRadius: 8, cursor: "pointer", border: `1px solid ${t.border}`,
            background: t.card, color: t.muted, fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center",
          }}>{dark ? "☀" : "🌙"}</button>
        </div>
      </nav>

      {/* ══ DETECT TAB ══ */}
      {tab === "detect" && (
        <>
          {/* Hero */}
          <section style={{ ...st.hero, background: t.surface }}>
            <div>
              <div style={{ fontSize: 9, letterSpacing: 4, color: t.accent, marginBottom: 12 }}>AI-POWERED SURFACE INSPECTION</div>
              <h1 style={{ fontFamily: "'Syne',sans-serif", fontSize: "clamp(32px,5vw,62px)", fontWeight: 800, lineHeight: 1.05, marginBottom: 16 }}>
                Metal Defect<br /><span style={{ color: t.accent }}>Detection System</span>
              </h1>
              <p style={{ color: t.muted, maxWidth: 500, lineHeight: 1.8, fontSize: 14, marginBottom: 28 }}>
                Hybrid CNN + Vision Transformer model classifying 7 surface defect types in real-time with per-class probability output and severity assessment.
              </p>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {[["7", "Defect Classes"], ["ViT Tiny", "Transformer"], ["CNN", "Extractor"], ["224×224", "Input Size"]].map(([v, l]) => (
                  <div key={l} style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "12px 18px", borderRadius: 10, background: t.card, border: `1px solid ${t.border}`, gap: 2 }}>
                    <span style={{ color: t.accent, fontWeight: 700, fontSize: 15 }}>{v}</span>
                    <span style={{ color: t.muted, fontSize: 9, letterSpacing: 1 }}>{l}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Carousel */}
            <div>
              <div style={{ width: 320, height: 220, borderRadius: 12, backgroundSize: "cover", backgroundPosition: "center", position: "relative", boxShadow: "0 24px 50px rgba(0,0,0,.5)", backgroundImage: `url(${CAROUSEL_ITEMS[carouselIdx].img})` }}>
                <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "28px 18px 16px", background: "linear-gradient(to top,rgba(0,0,0,.9),transparent)", borderRadius: "0 0 12px 12px", color: "#fff" }}>
                  <div style={{ fontSize: 9, letterSpacing: 3, color: t.accent }}>DEFECT TYPE</div>
                  <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 20, fontWeight: 800, marginTop: 4 }}>{CAROUSEL_ITEMS[carouselIdx].title}</div>
                  <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 3 }}>{DEFECT_META[CAROUSEL_ITEMS[carouselIdx].title]?.desc}</div>
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "center", gap: 6, marginTop: 10 }}>
                {CAROUSEL_ITEMS.map((_, i) => (
                  <span key={i} onClick={() => setCarouselIdx(i)} style={{ width: i === carouselIdx ? 18 : 6, height: 6, borderRadius: 3, background: i === carouselIdx ? t.accent : t.border, cursor: "pointer", transition: "width .3s" }} />
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 8 }}>
                {[["◀", -1], ["▶", 1]].map(([lbl, dir]) => (
                  <button key={lbl as string} onClick={() => setCarouselIdx(i => (i + (dir as number) + CAROUSEL_ITEMS.length) % CAROUSEL_ITEMS.length)}
                    style={{ width: 34, height: 34, borderRadius: 8, border: `1px solid ${t.border}`, background: t.card, color: t.muted, cursor: "pointer", fontSize: 13 }}>{lbl}</button>
                ))}
              </div>
            </div>
          </section>

          {/* Detection panel */}
          <section style={{ ...st.section, background: t.bg }}>
            <Tag>SINGLE IMAGE ANALYSIS</Tag>
            <div style={st.detectGrid}>
              {/* Drop zone */}
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div
                  className="zoom-img"
                  onDragOver={e => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                  onClick={() => !preview && fileInputRef.current?.click()}
                  style={{ borderRadius: 12, padding: 28, minHeight: 300, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "pointer", transition: "all .2s", background: t.card, border: `2px dashed ${dragging ? "#38bdf8" : t.border}` }}
                >
                  {preview ? (
                    <div style={{ position: "relative", width: "100%", display: "flex", justifyContent: "center" }}>
                      <img src={preview} style={{ maxWidth: "100%", maxHeight: 280, borderRadius: 8, objectFit: "contain" }} onClick={() => setZoomImg(preview)} />
                      {result?.gradcam_url && (
                        <div style={{ position: "absolute", top: 8, right: 8 }}>
                          <img src={result.gradcam_url} title="GradCAM Heatmap" style={{ width: 72, height: 72, borderRadius: 6, border: `2px solid ${t.accent}` }} />
                          <div style={{ fontSize: 8, textAlign: "center", color: t.accent, marginTop: 2 }}>GRADCAM</div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ textAlign: "center", color: t.muted }}>
                      <div style={{ fontSize: 36, marginBottom: 10 }}>⬆</div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>Drag & drop image</div>
                      <div style={{ fontSize: 11, marginTop: 6 }}>or click to browse · JPG PNG WEBP</div>
                    </div>
                  )}
                </div>
                <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => { if (e.target.files?.[0]) loadFile(e.target.files[0]); }} />
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={analyse} disabled={!imageFile || loading} style={{
                    flex: 1, padding: "13px 0", border: "none", borderRadius: 8,
                    fontFamily: "'Space Grotesk',sans-serif", fontWeight: 700, fontSize: 13, letterSpacing: 1.5,
                    background: (!imageFile || loading) ? t.surface : t.accent,
                    color: (!imageFile || loading) ? t.muted : "#000",
                    cursor: (!imageFile || loading) ? "not-allowed" : "pointer",
                  }}>
                    {loading ? <><span style={{ display: "inline-block", animation: "spin 1s linear infinite" }}>⟳</span> Analysing…</> : "▶  RUN ANALYSIS"}
                  </button>
                  {preview && (
                    <button onClick={() => { setImageFile(null); setPreview(null); setResult(null); }}
                      style={{ padding: "10px 16px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.card, color: t.muted, cursor: "pointer", fontSize: 12 }}>✕</button>
                  )}
                </div>
                {preview && (
                  <button onClick={() => fileInputRef.current?.click()} style={{ padding: "9px 0", borderRadius: 8, border: `1px dashed ${t.border}`, background: "transparent", color: t.muted, cursor: "pointer", fontSize: 12, fontFamily: "'Space Grotesk',sans-serif" }}>↺ Change Image</button>
                )}
              </div>

              {/* Result side */}
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {!result && !loading && (
                  <div style={{ borderRadius: 12, padding: 48, textAlign: "center", fontSize: 13, minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.card, border: `1px solid ${t.border}`, color: t.muted }}>
                    <div style={{ fontSize: 32, marginBottom: 12 }}>🔬</div>
                    Upload an image and click Run Analysis to see the detection result here.
                  </div>
                )}
                {loading && (
                  <div style={{ borderRadius: 12, padding: 48, textAlign: "center", fontSize: 13, minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.card, border: `1px solid ${t.border}`, color: t.muted, animation: "pulse 1.4s infinite" }}>
                    <div style={{ fontSize: 28, marginBottom: 10, animation: "spin 2s linear infinite", display: "inline-block" }}>⟳</div>
                    <div>Processing through CNN + ViT pipeline…</div>
                  </div>
                )}
                {result && (
                  <div style={{ animation: "fadeUp .4s ease" }}>
                    {/* Badge */}
                    <div style={{ borderRadius: 12, padding: "22px 24px", background: t.card, border: `1.5px solid ${DEFECT_META[result.defect]?.color}`, marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                          <div style={{ fontSize: 9, letterSpacing: 3, color: t.muted, marginBottom: 6 }}>DETECTED CLASS</div>
                          <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 26, fontWeight: 800, color: DEFECT_META[result.defect]?.color }}>
                            {result.defect === "Normal" ? "✓ NO DEFECT" : result.defect.toUpperCase()}
                          </div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: 30, fontWeight: 800, color: t.accent }}>{result.confidence}%</div>
                          <div style={{ fontSize: 9, color: t.muted, letterSpacing: 2 }}>CONFIDENCE</div>
                        </div>
                      </div>
                      <div style={{ fontSize: 12, color: t.muted, marginTop: 10, lineHeight: 1.6 }}>{DEFECT_META[result.defect]?.desc}</div>
                      <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                        <Chip label={`Severity: ${DEFECT_META[result.defect]?.severity}`} color={DEFECT_META[result.defect]?.color} />
                        <Chip label={`Action: ${DEFECT_META[result.defect]?.action}`} color={t.muted} />
                      </div>
                      <div style={{ marginTop: 10, fontSize: 11, color: t.muted }}>
                        <span style={{ color: t.accent }}>⚙ Cause: </span>{DEFECT_META[result.defect]?.cause}
                      </div>
                    </div>
                    {/* Prob bars */}
                    <div style={{ borderRadius: 12, padding: "18px 22px", background: t.card, border: `1px solid ${t.border}` }}>
                      <div style={{ fontSize: 9, letterSpacing: 3, color: t.accent, marginBottom: 14 }}>ALL CLASS PROBABILITIES</div>
                      {CLASSES.map(cls => {
                        const prob = result.all_probs?.[cls] ?? (cls === result.defect ? result.confidence : 0);
                        const isTop = cls === result.defect;
                        return (
                          <div key={cls} style={{ marginBottom: 9 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3, color: isTop ? t.text : t.muted }}>
                              <span style={{ fontWeight: isTop ? 700 : 400 }}>{DEFECT_META[cls]?.icon} {cls}</span>
                              <span style={{ color: isTop ? DEFECT_META[cls].color : t.muted }}>{prob.toFixed(2)}%</span>
                            </div>
                            <div style={{ height: 5, background: t.border, borderRadius: 3, overflow: "hidden" }}>
                              <div className="bar" style={{ height: "100%", borderRadius: 3, width: `${Math.min(prob, 100)}%`, background: isTop ? DEFECT_META[cls].color : t.muted, opacity: isTop ? 1 : .5 }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Architecture */}
            <div style={{ marginTop: 52 }}>
              <Tag>MODEL ARCHITECTURE</Tag>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", justifyContent: "center", marginTop: 16 }}>
                {[
                  { icon: "🖼", label: "Input",      desc: "224×224 RGB" },
                  { icon: "⚙", label: "CNN Block",  desc: "Conv→ReLU→Pool ×2" },
                  { icon: "🔁", label: "Adapter",    desc: "1×1 Conv + Resize" },
                  { icon: "🤖", label: "ViT Tiny",   desc: "Patch16 Transformer" },
                  { icon: "📊", label: "Classifier", desc: "7-class Softmax" },
                ].map((step, i, arr) => (
                  <div key={step.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ borderRadius: 10, padding: "14px 12px", textAlign: "center", width: 110, background: t.card, border: `1px solid ${t.border}` }}>
                      <div style={{ fontSize: 20 }}>{step.icon}</div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: t.accent, marginTop: 5, letterSpacing: .5 }}>{step.label}</div>
                      <div style={{ fontSize: 9, color: t.muted, marginTop: 3 }}>{step.desc}</div>
                    </div>
                    {i < arr.length - 1 && <span style={{ color: t.muted, fontSize: 16 }}>→</span>}
                  </div>
                ))}
              </div>
            </div>
          </section>
        </>
      )}

      {/* ══ BATCH TAB ══ */}
      {tab === "batch" && (
        <section style={{ ...st.section, background: t.bg }}>
          <Tag>BATCH INSPECTION</Tag>
          <h2 style={{ ...st.h2, color: t.text }}>Multi-Image Analysis</h2>
          <p style={{ color: t.muted, fontSize: 13, marginBottom: 24, lineHeight: 1.8 }}>
            Upload multiple surface images. Each will be processed sequentially and added to your inspection history.
          </p>
          <div style={{ display: "flex", gap: 10, marginBottom: 24, flexWrap: "wrap" }}>
            <button onClick={() => batchInputRef.current?.click()} style={{ padding: "10px 16px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.card, color: t.text, cursor: "pointer", fontSize: 12, fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600 }}>
              + Add Images ({batchFiles.length} selected)
            </button>
            <input ref={batchInputRef} type="file" accept="image/*" multiple style={{ display: "none" }} onChange={e => { if (e.target.files) setBatchFiles(Array.from(e.target.files)); }} />
            <button onClick={runBatch} disabled={!batchFiles.length || batchRunning} style={{
              padding: "10px 20px", border: "none", borderRadius: 8, fontFamily: "'Space Grotesk',sans-serif", fontWeight: 700, fontSize: 12, letterSpacing: 1,
              background: (!batchFiles.length || batchRunning) ? t.surface : t.accent,
              color: (!batchFiles.length || batchRunning) ? t.muted : "#000",
              cursor: (!batchFiles.length || batchRunning) ? "not-allowed" : "pointer",
            }}>
              {batchRunning ? "⟳ Processing…" : "▶ Run Batch"}
            </button>
            {batchResults.length > 0 && (
              <button onClick={() => exportCSV(batchResults.filter(r => r.status === "done") as HistoryEntry[])}
                style={{ padding: "10px 16px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.card, color: t.accent, cursor: "pointer", fontSize: 12, fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600 }}>
                ↓ Export CSV
              </button>
            )}
            {batchResults.length > 0 && !batchRunning && (
              <button onClick={() => { setBatchFiles([]); setBatchResults([]); }}
                style={{ padding: "10px 16px", borderRadius: 8, border: `1px solid ${t.border}`, background: "transparent", color: t.muted, cursor: "pointer", fontSize: 12 }}>
                ✕ Clear
              </button>
            )}
          </div>

          {batchResults.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(170px,1fr))", gap: 14 }}>
              {batchResults.map((r, i) => (
                <div key={i} style={{ borderRadius: 10, overflow: "hidden", background: t.card, border: `1px solid ${r.status === "error" ? "#ef4444" : r.status === "done" ? (DEFECT_META[r.defect]?.color ?? t.border) : t.border}` }}>
                  <img src={r.preview} style={{ width: "100%", height: 110, objectFit: "cover", cursor: "zoom-in" }} onClick={() => setZoomImg(r.preview)} />
                  <div style={{ padding: "10px 12px" }}>
                    <div style={{ fontSize: 10, color: t.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 4 }}>{r.filename}</div>
                    {r.status === "pending" && <div style={{ fontSize: 11, color: t.muted, animation: "pulse 1.5s infinite" }}>Pending…</div>}
                    {r.status === "error"   && <div style={{ fontSize: 11, color: "#ef4444" }}>⚠ Error</div>}
                    {r.status === "done"    && (
                      <>
                        <div style={{ fontSize: 13, fontWeight: 700, color: DEFECT_META[r.defect]?.color }}>{r.defect}</div>
                        <div style={{ fontSize: 11, color: t.muted }}>{r.confidence}% confidence</div>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {!batchFiles.length && !batchResults.length && (
            <div style={{ borderRadius: 12, padding: 48, textAlign: "center", fontSize: 13, minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.card, border: `1px solid ${t.border}`, color: t.muted, maxWidth: 500 }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📂</div>
              Select multiple images above to run a batch inspection.
            </div>
          )}
        </section>
      )}

      {/* ══ HISTORY TAB ══ */}
      {tab === "history" && (
        <section style={{ ...st.section, background: t.bg }}>
          <Tag>INSPECTION LOG</Tag>
          <h2 style={{ ...st.h2, color: t.text }}>Detection History</h2>
          <div style={{ display: "flex", gap: 10, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
            <select value={filterDefect} onChange={e => setFilterDefect(e.target.value)}
              style={{ padding: "8px 12px", borderRadius: 8, fontSize: 12, background: t.card, border: `1px solid ${t.border}`, color: t.text, fontFamily: "'Space Grotesk',sans-serif", outline: "none", cursor: "pointer" }}>
              {["All", ...CLASSES].map(c => <option key={c}>{c}</option>)}
            </select>
            {sortedHistory.length > 0 && (
              <>
                <button onClick={() => exportCSV(sortedHistory)} style={{ padding: "8px 14px", borderRadius: 8, border: `1px solid ${t.border}`, background: t.card, color: t.accent, cursor: "pointer", fontSize: 12, fontFamily: "'Space Grotesk',sans-serif", fontWeight: 600 }}>↓ Export CSV</button>
                <button onClick={() => setHistory([])} style={{ padding: "8px 14px", borderRadius: 8, border: `1px solid ${t.border}`, background: "transparent", color: "#ef4444", cursor: "pointer", fontSize: 12 }}>✕ Clear All</button>
              </>
            )}
            <span style={{ fontSize: 11, color: t.muted, marginLeft: "auto" }}>{sortedHistory.length} records</span>
          </div>

          {sortedHistory.length === 0 ? (
            <div style={{ borderRadius: 12, padding: 48, textAlign: "center", fontSize: 13, minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.card, border: `1px solid ${t.border}`, color: t.muted }}>
              <div style={{ fontSize: 32, marginBottom: 10 }}>📋</div>
              No inspections yet. Run analyses to build your log.
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, color: t.text }}>
                <thead>
                  <tr style={{ background: t.surface }}>
                    {(["id", "preview", "filename", "defect", "confidence", "severity", "timestamp"] as (keyof HistoryEntry | "preview")[]).map((h, i) => (
                      <th key={h} onClick={() => i > 2 && toggleSort(h as keyof HistoryEntry)} style={{ padding: "12px 16px", textAlign: "left", fontSize: 9, letterSpacing: 2, fontWeight: 700, color: t.accent, cursor: i > 2 ? "pointer" : "default" }}>
                        {String(h).charAt(0).toUpperCase() + String(h).slice(1)}
                        {sortField === h ? (sortAsc ? " ↑" : " ↓") : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedHistory.map(row => (
                    <tr key={row.id} className="hist-row" style={{ borderBottom: `1px solid ${t.border}` }}>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle" }}>{row.id}</td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle" }}>
                        <img src={row.preview} style={{ width: 40, height: 40, objectFit: "cover", borderRadius: 4, cursor: "zoom-in" }} onClick={() => setZoomImg(row.preview)} />
                      </td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.filename}</td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle", color: DEFECT_META[row.defect]?.color, fontWeight: 600 }}>{row.defect}</td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 50, height: 4, background: t.border, borderRadius: 2, overflow: "hidden" }}>
                            <div className="bar" style={{ height: "100%", width: `${row.confidence}%`, background: DEFECT_META[row.defect]?.color }} />
                          </div>
                          {row.confidence}%
                        </div>
                      </td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle" }}>
                        <Chip label={row.severity} color={DEFECT_META[row.defect]?.color} small />
                      </td>
                      <td style={{ padding: "11px 16px", verticalAlign: "middle", color: t.muted, fontSize: 11 }}>{row.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* ══ STATS TAB ══ */}
      {tab === "stats" && (
        <section style={{ ...st.section, background: t.bg }}>
          <Tag>ANALYTICS DASHBOARD</Tag>
          <h2 style={{ ...st.h2, color: t.text }}>Inspection Statistics</h2>

          {history.length === 0 ? (
            <div style={{ borderRadius: 12, padding: 48, textAlign: "center", fontSize: 13, minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.card, border: `1px solid ${t.border}`, color: t.muted }}>
              <div style={{ fontSize: 32, marginBottom: 10 }}>📊</div>
              Run some inspections to see statistics here.
            </div>
          ) : (
            <>
              {/* KPIs */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 14, marginBottom: 28 }}>
                {[
                  { label: "Total Inspections", value: total,         color: t.accent  },
                  { label: "Defects Found",      value: defects,       color: "#f97316" },
                  { label: "Pass Rate",          value: `${passRate}%`, color: "#10b981" },
                  { label: "Avg Confidence",     value: `${avgConf}%`,  color: "#a78bfa" },
                  { label: "Top Defect",         value: topDefect,     color: "#f43f5e" },
                ].map(kpi => (
                  <div key={kpi.label} style={{ borderRadius: 12, padding: "20px 18px", background: t.card, border: `1px solid ${t.border}` }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: kpi.color }}>{kpi.value}</div>
                    <div style={{ fontSize: 9, color: t.muted, marginTop: 4, letterSpacing: 1 }}>{kpi.label.toUpperCase()}</div>
                  </div>
                ))}
              </div>

              {/* Distribution chart */}
              <div style={{ borderRadius: 12, padding: "22px 24px", background: t.card, border: `1px solid ${t.border}`, marginBottom: 14 }}>
                <div style={{ fontSize: 9, letterSpacing: 3, color: t.accent, marginBottom: 18 }}>DEFECT CLASS DISTRIBUTION</div>
                {CLASSES.map(cls => {
                  const count = freqMap[cls] ?? 0;
                  const pct   = total ? (count / total) * 100 : 0;
                  return (
                    <div key={cls} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4, color: count > 0 ? t.text : t.muted }}>
                        <span>{DEFECT_META[cls]?.icon} {cls}</span>
                        <span style={{ color: DEFECT_META[cls]?.color }}>{count} ({pct.toFixed(1)}%)</span>
                      </div>
                      <div style={{ height: 8, background: t.border, borderRadius: 4, overflow: "hidden" }}>
                        <div className="bar" style={{ height: "100%", borderRadius: 4, width: `${pct}%`, background: DEFECT_META[cls]?.color, opacity: count ? 1 : .3 }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Severity grid */}
              <div style={{ borderRadius: 12, padding: "22px 24px", background: t.card, border: `1px solid ${t.border}` }}>
                <div style={{ fontSize: 9, letterSpacing: 3, color: t.accent, marginBottom: 18 }}>SEVERITY BREAKDOWN</div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {(["None", "Low", "Medium", "High", "Critical"] as const).map(sev => {
                    const cnt = history.filter(r => DEFECT_META[r.defect]?.severity === sev).length;
                    const col = ({ None: "#10b981", Low: "#60a5fa", Medium: "#eab308", High: "#f97316", Critical: "#f43f5e" })[sev];
                    return (
                      <div key={sev} style={{ flex: 1, minWidth: 80, textAlign: "center", padding: "16px 12px", borderRadius: 10, background: col + "18", border: `1px solid ${col}44` }}>
                        <div style={{ fontSize: 24, fontWeight: 800, color: col }}>{cnt}</div>
                        <div style={{ fontSize: 9, color: t.muted, marginTop: 4, letterSpacing: 1 }}>{sev.toUpperCase()}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {/* ══ ZOOM MODAL ══ */}
      {zoomImg && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.88)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 999, backdropFilter: "blur(8px)" }} onClick={() => setZoomImg(null)}>
          <div style={{ position: "relative" }} onClick={e => e.stopPropagation()}>
            <img src={zoomImg} style={{ maxWidth: "90vw", maxHeight: "85vh", borderRadius: 10, objectFit: "contain" }} />
            <button onClick={() => setZoomImg(null)} style={{ position: "absolute", top: 10, right: 10, width: 32, height: 32, borderRadius: 8, border: "none", background: "rgba(0,0,0,.6)", color: "#fff", fontSize: 16, cursor: "pointer" }}>✕</button>
          </div>
        </div>
      )}

      {/* ══ FOOTER ══ */}
      <footer style={{ padding: "20px 40px", textAlign: "center", fontSize: 11, letterSpacing: .5, background: t.surface, borderTop: `1px solid ${t.border}`, color: t.muted }}>
        <span style={{ color: t.accent, fontFamily: "'Syne',sans-serif", fontWeight: 800 }}>MetalInspect</span>
        {" "}© 2026 · CNN + ViT Metallic Surface Defect Detection
      </footer>
    </div>
  );
}

/* ─── Helpers ─── */
function Tag({ children }: { children: string }) {
  return <div style={{ fontSize: 9, letterSpacing: 4, color: "#38bdf8", marginBottom: 10 }}>{children}</div>;
}
function Chip({ label, color, small }: { label: string; color: string; small?: boolean }) {
  return (
    <span style={{ fontSize: small ? 9 : 10, padding: small ? "2px 8px" : "4px 11px", borderRadius: 20, background: color + "20", color, fontWeight: 600, letterSpacing: .5 }}>{label}</span>
  );
}

/* ─── Static styles ─── */
const st: Record<string, CSSProperties> = {
  page:        { minHeight: "100vh", fontFamily: "'Space Grotesk',sans-serif", lineHeight: 1.5 },
  nav:         { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 40px", position: "sticky", top: 0, zIndex: 200, backdropFilter: "blur(14px)" },
  hero:        { display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 40, padding: "64px 40px 56px", flexWrap: "wrap" },
  section:     { padding: "56px 40px", maxWidth: 1200, margin: "0 auto" },
  h2:          { fontFamily: "'Syne',sans-serif", fontSize: 28, fontWeight: 800, marginBottom: 24, letterSpacing: .5 },
  detectGrid:  { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 },
};
