import React, { useState, useCallback, useRef } from 'react';
import UploadZone from './UploadZone';
import RecoveryViewer from './RecoveryViewer';
import CorruptionReport from './CorruptionReport';
import { fetchWithRetry } from '../utils/api';

// ─── Shared method config ──────────────────────────────────────────────────────
const METHODS = [
  { key: 'corrupted', label: 'Input',    tag: 'Corrupted',    desc: 'Raw corrupted upload - unmodified',                               color: 'var(--red-bright)',    bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.25)',  icon: '📤' },
  { key: 'opencv',    label: 'OpenCV',   tag: 'Traditional',  desc: 'NLMeans + Bilateral + Unsharp - pure signal processing, no AI',   color: 'var(--cyan)',          bg: 'rgba(0,245,255,0.08)', border: 'rgba(0,245,255,0.25)', icon: '⚙️' },
  { key: 'ai',        label: 'AI Model', tag: 'Neural Net',   desc: 'U-Net autoencoder inpainting - learns anatomical context',        color: 'var(--amber-bright)', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.25)', icon: '🤖', requiresAI: true },
  { key: 'final',     label: 'Hybrid',   tag: 'Best Result',  desc: 'Full pipeline - OpenCV + AI + targeted inpainting blend',        color: 'var(--green-bright)', bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.25)', icon: '✨' },
];

// Default wipe pairs (left vs right)
const WIPE_PAIRS = [
  { left: 'opencv',    right: 'final',    label: 'OpenCV vs Hybrid' },
  { left: 'corrupted', right: 'final',    label: 'Input vs Hybrid' },
  { left: 'opencv',    right: 'ai',       label: 'OpenCV vs AI' },
  { left: 'corrupted', right: 'opencv',   label: 'Input vs OpenCV' },
];

function getMeta(key) { return METHODS.find(m => m.key === key) || METHODS[3]; }
function b64Src(b64) { return b64 ? `data:image/png;base64,${b64}` : null; }

// ─── Wipe Viewer ──────────────────────────────────────────────────────────────
// singleKey: when set, shows a single image fullframe instead of the wipe comparison
function WipeViewer({ imgMap, pairIdx, aiAvailable, singleKey }) {
  // Single-image view
  if (singleKey) {
    const m    = getMeta(singleKey);
    const src  = b64Src(imgMap[singleKey]);
    const unavail = m.requiresAI && !aiAvailable;
    return (
      <div className="wipe-root">
        <div className="wipe-frame" style={{ cursor: 'default' }}>
          {src && !unavail
            ? <img src={src} className="wipe-img" alt={m.label} draggable={false} />
            : <div className="wipe-unavail">{m.icon} {m.tag} — Not available</div>
          }
          <div className="wipe-label wipe-label-left" style={{ color: m.color, background: m.bg, borderColor: m.border }}>
            {m.icon} {m.tag}
          </div>
        </div>
      </div>
    );
  }

  // Wipe comparison view
  const [wipePos, setWipePos]   = useState(50);
  const [dragging, setDragging] = useState(false);
  const frameRef = useRef(null);

  const pair = WIPE_PAIRS[pairIdx] || WIPE_PAIRS[0];
  const leftKey  = pair.left;
  const rightKey = pair.right;
  const leftMeta  = getMeta(leftKey);
  const rightMeta = getMeta(rightKey);
  const leftSrc   = b64Src(imgMap[leftKey]);
  const rightSrc  = b64Src(imgMap[rightKey]);

  const updatePos = useCallback((clientX) => {
    if (!frameRef.current) return;
    const { left, width } = frameRef.current.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((clientX - left) / width) * 100));
    setWipePos(pct);
  }, []);

  const onMouseMove  = useCallback(e => { if (dragging) updatePos(e.clientX); }, [dragging, updatePos]);
  const onTouchMove  = useCallback(e => updatePos(e.touches[0].clientX), [updatePos]);
  const onMouseDown  = useCallback(e => { setDragging(true); updatePos(e.clientX); }, [updatePos]);
  const onMouseUp    = useCallback(() => setDragging(false), []);

  // Disable AI pair if AI unavailable
  const leftUnavail  = METHODS.find(m => m.key === leftKey)?.requiresAI  && !aiAvailable;
  const rightUnavail = METHODS.find(m => m.key === rightKey)?.requiresAI && !aiAvailable;

  return (
    <div className="wipe-root">
      <div
        ref={frameRef}
        className="wipe-frame"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchMove={onTouchMove}
        style={{ cursor: dragging ? 'col-resize' : 'ew-resize', userSelect: 'none' }}
        aria-label="Drag to compare recovery methods"
      >
        {/* LEFT image — full width, clipped on right */}
        {leftSrc && !leftUnavail
          ? <img src={leftSrc} className="wipe-img wipe-img-left" alt={leftMeta.label} draggable={false} />
          : <div className="wipe-unavail">{leftMeta.icon} {leftMeta.tag} N/A</div>
        }

        {/* RIGHT image — clipped on left from wipePos */}
        {rightSrc && !rightUnavail
          ? <img
              src={rightSrc}
              className="wipe-img wipe-img-right"
              alt={rightMeta.label}
              draggable={false}
              style={{ clipPath: `inset(0 0 0 ${wipePos}%)` }}
            />
          : <div className="wipe-unavail wipe-unavail-right" style={{ clipPath: `inset(0 0 0 ${wipePos}%)` }}>
              {rightMeta.icon} {rightMeta.tag} N/A
            </div>
        }

        {/* Slider line + handle */}
        <div className="wipe-line" style={{ left: `${wipePos}%` }}>
          <div className="wipe-handle">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
              <polyline points="15 18 9 12 15 6"/><polyline points="9 18 3 12 9 6"/><polyline points="15 18 21 12 15 6" transform="translate(0,0) scale(-1,1) translate(-24,0)"/>
            </svg>
          </div>
        </div>

        {/* Corner labels */}
        <div className="wipe-label wipe-label-left" style={{ color: leftMeta.color, background: leftMeta.bg, borderColor: leftMeta.border }}>
          {leftMeta.icon} {leftMeta.tag}
        </div>
        <div className="wipe-label wipe-label-right" style={{ color: rightMeta.color, background: rightMeta.bg, borderColor: rightMeta.border }}>
          {rightMeta.tag} {rightMeta.icon}
        </div>
      </div>
    </div>
  );
}

// ─── Method Selector (right panel) ────────────────────────────────────────────
// ─── Right panel: Single View only ───────────────────────────────────────────
function MethodSelector({ singleKey, setSingleKey, aiAvailable }) {
  const handleSingle = (key) => {
    setSingleKey(prev => prev === key ? null : key);
  };
  return (
    <div className="ms-root">
      <div className="ms-heading">Single View</div>
      {METHODS.map(m => {
        const disabled = m.requiresAI && !aiAvailable;
        const active   = singleKey === m.key;
        return (
          <button
            key={m.key}
            className={`ms-method-row${disabled ? ' disabled' : ''}${active ? ' active' : ''}`}
            style={{
              borderColor: active ? m.border : disabled ? 'transparent' : 'rgba(255,255,255,0.06)',
              background:  active ? m.bg : undefined,
              cursor: disabled ? 'not-allowed' : 'pointer',
              width: '100%', textAlign: 'left',
            }}
            onClick={() => !disabled && handleSingle(m.key)}
            disabled={disabled}
            id={`ms-single-${m.key}`}
            title={disabled ? 'Not available' : `View ${m.label} only`}
          >
            <span className="ms-method-icon" style={{ color: m.color }}>{m.icon}</span>
            <div className="ms-method-text">
              <span className="ms-method-label" style={{ color: m.color }}>{m.label}</span>
              <span className="ms-method-sub">{disabled ? 'Not available' : m.desc.slice(0, 44) + '…'}</span>
            </div>
            {active && <div className="ms-active-bar" />}
          </button>
        );
      })}
    </div>
  );
}

// ─── Below-image: Compare Pairs horizontal strip ──────────────────────────────
function ComparePairsBar({ pairIdx, setPairIdx, singleKey, setSingleKey, aiAvailable }) {
  const handlePair = (i) => { setPairIdx(i); setSingleKey(null); };
  return (
    <div className="cpb-root">
      <span className="cpb-heading">Compare Pairs</span>
      <div className="cpb-row">
        {WIPE_PAIRS.map((pair, i) => {
          const lm = getMeta(pair.left);
          const rm = getMeta(pair.right);
          const disabled = (METHODS.find(m => m.key === pair.left)?.requiresAI || METHODS.find(m => m.key === pair.right)?.requiresAI) && !aiAvailable;
          const active = pairIdx === i && !singleKey;
          return (
            <button
              key={i}
              className={`cpb-card${active ? ' active' : ''}${disabled ? ' disabled' : ''}`}
              onClick={() => !disabled && handlePair(i)}
              disabled={disabled}
              id={`cpb-pair-${i}`}
              title={disabled ? 'AI unavailable' : pair.label}
            >
              <span className="cpb-badge" style={{ color: lm.color, borderColor: lm.border, background: lm.bg }}>{lm.icon} {lm.tag}</span>
              <span className="cpb-vs">vs</span>
              <span className="cpb-badge" style={{ color: rm.color, borderColor: rm.border, background: rm.bg }}>{rm.icon} {rm.tag}</span>
              {active && <div className="cpb-active-bar" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}


function MethodComparisonViewer({ images, aiAvailable, corrupted }) {
  const [active, setActive] = useState('final');

  // Build a unified image map including the raw corrupted upload
  const imgMap = { ...(images || {}), corrupted };

  const activeMethod = METHODS.find(m => m.key === active) || METHODS[3];
  const activeImg    = imgMap[active];

  return (
    <div className="mcv-root">
      {/* Method selector tabs */}
      <div className="mcv-tabs" role="tablist" aria-label="Recovery method selector">
        {METHODS.map(m => {
          const disabled = m.requiresAI && !aiAvailable;
          const isActive = active === m.key;
          return (
            <button
              key={m.key}
              role="tab"
              aria-selected={isActive}
              aria-disabled={disabled}
              className={`mcv-tab${isActive ? ' active' : ''}${disabled ? ' disabled' : ''}`}
              style={isActive ? { '--tab-color': m.color, '--tab-bg': m.bg, '--tab-border': m.border } : {}}
              onClick={() => !disabled && setActive(m.key)}
              id={`mcv-tab-${m.key}`}
              title={disabled ? 'AI model not available for this file' : m.desc}
            >
              <span className="mcv-tab-icon" aria-hidden="true">{m.icon}</span>
              <span className="mcv-tab-label">{m.label}</span>
              <span
                className="mcv-tab-tag"
                style={isActive ? { color: m.color, borderColor: m.border, background: m.bg } : {}}
              >
                {disabled ? 'N/A' : m.tag}
              </span>
            </button>
          );
        })}
      </div>

      {/* Image frame */}
      <div className="mcv-frame" style={{ '--frame-border': activeMethod.border }}>
        {activeImg
          ? <img
              src={`data:image/png;base64,${activeImg}`}
              alt={`${activeMethod.label} recovery output`}
              className="mcv-image"
            />
          : <div className="mcv-placeholder">
              <span className="mcv-placeholder-icon" aria-hidden="true">🚫</span>
              <span>Not available — AI model was not applied to this file</span>
            </div>
        }
        {/* Floating method label */}
        <div className="mcv-overlay-label" style={{ color: activeMethod.color, background: activeMethod.bg, borderColor: activeMethod.border }}>
          {activeMethod.icon} {activeMethod.tag}
        </div>
      </div>

      {/* Description row */}
      <div className="mcv-desc-row">
        <div className="mcv-desc-icon" style={{ color: activeMethod.color }} aria-hidden="true">{activeMethod.icon}</div>
        <div className="mcv-desc-text">
          <span className="mcv-desc-label" style={{ color: activeMethod.color }}>{activeMethod.label} — {activeMethod.tag}</span>
          <span className="mcv-desc-detail">{activeMethod.desc}</span>
        </div>
      </div>
    </div>
  );
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PIPELINE_STEPS = [
  { id: 'load',        icon: '⚙️',  label: 'Initializing Engine',       desc: 'Recovery pipeline ready' },
  { id: 'analyze',    icon: '🔎',  label: 'Detecting Corruption',       desc: 'Severity and pattern analysis' },
  { id: 'reconstruct',icon: '🤖',  label: 'Reconstructing Image',       desc: 'Structure-aware repair' },
  { id: 'inpaint',    icon: '🖌️', label: 'Inpainting Damaged Regions', desc: 'Pixel region fill' },
  { id: 'metadata',   icon: '🏷️', label: 'Restoring Metadata',         desc: 'Tag restoration' },
  { id: 'package',    icon: '📦',  label: 'Packaging Output',           desc: 'Recovered DICOM export' },
];

export default function RecoverWorkspace() {
  const [file, setFile] = useState(null);
  const [step, setStep] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [wakingUp, setWakingUp] = useState(false); // Render cold-start indicator
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showMeta, setShowMeta] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [pairIdx, setPairIdx] = useState(0);   // active wipe pair index
  const [singleKey, setSingleKey] = useState(null); // active single-view method (null = wipe mode)

  const handleRecover = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setStep(0);
    setShowMeta(false);
    setShowReport(false);
    setPairIdx(0);
    setSingleKey(null);

    const timers = PIPELINE_STEPS.map((_, i) => setTimeout(() => setStep(i), i * 900));
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetchWithRetry(
        `${API_BASE}/recover`,
        {
          method: 'POST',
          body: fd,
          onRetry: (attempt) => {
            // Show waking-up banner — Render free tier cold-starts take 30-60s
            if (attempt === 1) setWakingUp(true);
          },
        },
        2  // retry up to 2 more times (total 3 attempts)
      );
      setWakingUp(false);
      timers.forEach(clearTimeout);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Recovery failed');
      }
      const data = await res.json();
      setStep(PIPELINE_STEPS.length);
      setResult(data);
    } catch (e) {
      timers.forEach(clearTimeout);
      setWakingUp(false);
      setStep(-1);
      setError(
        e.name === 'TypeError' || e.message?.includes('fetch')
          ? 'Cannot reach the backend. The server may be starting up — please wait 30 seconds and try again.'
          : e.message || 'Recovery failed. Is the backend running?'
      );
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleReset = () => {
    setResult(null); setError(null); setStep(-1);
    setFile(null); setShowMeta(false); setShowReport(false); setPairIdx(0); setSingleKey(null);
  };

  const handleDownload = () => {
    if (!result?.recovered_dicom_b64) return;
    const bytes = atob(result.recovered_dicom_b64);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    const blob = new Blob([arr], { type: 'application/dicom' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `recovered_${file?.name || 'dicom.dcm'}`; a.click();
    URL.revokeObjectURL(url);
  };

  const progressPct = loading && step >= 0 ? ((step + 1) / PIPELINE_STEPS.length) * 100 : 0;
  const restoredTags = result?.restored_metadata ? Object.entries(result.restored_metadata) : [];
  const corruption = result?.corruption_report || null;
  const affectedPct = corruption?.affected_percentage ?? 0;
  const statusLabel = corruption?.recoverable ? 'RECOVERY OUTPUT READY' : 'LIMITED RECOVERY';
  const statusTone  = corruption?.recoverable ? 'var(--green-bright)' : 'var(--amber-bright)';
  const summaryLine = corruption
    ? `${corruption.type} · ${affectedPct}% affected · ${corruption.recoverable ? 'Recoverable' : 'Severe'}`
    : 'Approximate reconstruction generated from detected corruption patterns';

  // Build unified image map for wipe viewer
  const imgMap = result ? {
    ...(result.method_images || {}),
    corrupted: result.corrupted_image,
  } : {};

  return (
    <div className="fw-workspace" id="recover-workspace">
      <div className="fw-topbar">
        <div className="fw-topbar-left">
          <div className="fw-topbar-icon recover-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          </div>
          <div>
            <h1 className="fw-topbar-title rw-title">AI Recovery Engine</h1>
            <p className="fw-topbar-sub">Approximate reconstruction · OpenCV inpainting · Metadata restoration</p>
          </div>
        </div>
        <div className="fw-topbar-right">
          <div className="fw-status-pill rw-status-pill">
            <span className="fw-status-pulse rw-pulse" aria-hidden="true" />
            <span>Recovery Engine Active</span>
          </div>
          {result && (
            <>
              <button className="fw-action-btn ghost" id="recover-new-btn" onClick={handleReset}>↺ New Recovery</button>
              <button className="fw-action-btn rw-primary" id="recover-download-btn"
                disabled={!result?.recovered_dicom_b64} onClick={handleDownload}>
                ↓ Download DICOM
              </button>
            </>
          )}
        </div>
      </div>

      <div className="fw-content">
        {!result && !loading && (
          <div className="fw-idle">
            <div className="fw-upload-hero-block rw-hero-block">
              <div className="fw-step-label">STEP 1 — UPLOAD CORRUPTED FILE</div>
              <UploadZone id="recover-upload-zone" onFileSelect={setFile} accept=".dcm,.DCM" label="Upload corrupted DICOM file" />
              {error && <div className="fw-error-bar" role="alert" aria-live="polite"><span aria-hidden="true">⚠️ </span> {error}</div>}
              <button className={`fw-cta-btn rw-cta-btn${!file ? ' fw-cta-disabled' : ''}`} id="recover-start-btn"
                onClick={handleRecover} disabled={!file} aria-label="Start AI recovery">
                <span className="fw-cta-glow rw-cta-glow" aria-hidden="true" />
                <span className="fw-cta-icon" aria-hidden="true">⚡</span>
                <span className="fw-cta-label rw-cta-label">Start AI Recovery</span>
                <span className="fw-cta-hint">{file ? file.name : 'Upload a corrupted .dcm file'}</span>
              </button>
            </div>
            <div className="fw-pipeline-preview">
              <div className="fw-step-label">RECOVERY PIPELINE — 6 STEPS</div>
              <div className="fw-steps-row rw-steps-row">
                {PIPELINE_STEPS.map((s, i) => (
                  <div key={s.id} className="fw-step-card fw-step-idle">
                    <span className="fw-step-num-badge rw-step-num">{i + 1}</span>
                    <span className="fw-step-icon">{s.icon}</span>
                    <span className="fw-step-name">{s.label}</span>
                    <span className="fw-step-desc">{s.desc}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rw-expect-row">
              {[
                { icon: '🤖',  label: 'Approximate Repair',   desc: 'Model-assisted reconstruction' },
                { icon: '🖌️', label: 'Targeted Inpainting',  desc: 'Damaged region fill' },
                { icon: '🏷️', label: 'Metadata Restoration', desc: 'Tag-level recovery' },
                { icon: '⚠️', label: 'Human Review Needed',  desc: 'Not primary diagnostic output' },
              ].map(item => (
                <div key={item.label} className="rw-expect-card">
                  <span className="rw-expect-icon" aria-hidden="true">{item.icon}</span>
                  <span className="rw-expect-label">{item.label}</span>
                  <span className="rw-expect-desc">{item.desc}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="fw-processing">
            {wakingUp && (
              <div className="fw-wakeup-banner" role="status" aria-live="polite">
                <span className="fw-wakeup-icon" aria-hidden="true">🌙</span>
                <span>Server is waking up (free tier cold-start) — retrying automatically…</span>
              </div>
            )}
            <div className="fw-proc-header">
              <div className="fw-proc-orbs" aria-hidden="true">
                <div className="fw-orb rw-orb-a" /><div className="fw-orb rw-orb-b" />
              </div>
              <div className="fw-proc-title">AI Recovery Processing</div>
              <div className="fw-proc-file rw-proc-file" aria-live="polite">{file?.name}</div>
            </div>
            <div className="fw-steps-row fw-steps-active rw-steps-active" role="list">
              {PIPELINE_STEPS.map((s, i) => {
                const state = i < step ? 'done' : i === step ? 'active' : 'pending';
                return (
                  <div key={s.id} className={`fw-step-card fw-step-${state} rw-step-${state}`} role="listitem">
                    <div className="fw-step-indicator rw-step-indicator" aria-hidden="true">
                      {state === 'done'    && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>}
                      {state === 'active'  && <span className="rw-step-spinner" />}
                      {state === 'pending' && <span className="fw-step-num">{i + 1}</span>}
                    </div>
                    <span className="fw-step-icon">{s.icon}</span>
                    <div className="fw-step-text">
                      <span className="fw-step-name">{s.label}</span>
                      <span className="fw-step-desc">{s.desc}</span>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="fw-progress-wrap">
              <div className="fw-progress-track">
                <div className="rw-progress-fill" style={{ width: `${progressPct}%` }}
                  role="progressbar" aria-valuenow={progressPct} aria-valuemin={0} aria-valuemax={100} />
              </div>
              <div className="fw-progress-label" aria-live="polite">{PIPELINE_STEPS[step]?.label || 'Initializing…'}</div>
            </div>
            <div className="fw-proc-wave" aria-hidden="true">
              {Array.from({ length: 24 }, (_, i) => (
                <div key={i} className="fw-wave-bar rw-wave-bar" style={{ animationDelay: `${i * 0.06}s` }} />
              ))}
            </div>
          </div>
        )}

        {result && !loading && (
          <div className="fw-results rw-results">
            {/* Verdict strip */}
            <div className="fw-verdict-strip rw-verdict-strip">
              <div className="fvs-left">
                <span className="status-badge original">{statusLabel}</span>
                <span className="fvs-conf">{summaryLine}</span>
              </div>
              <div className="fvs-right">
                {restoredTags.length > 0 && (
                  <button className="fvs-btn" id="recover-meta-btn" onClick={() => setShowMeta(v => !v)}>
                    {showMeta ? '▲ Hide Metadata' : `🏷️ ${restoredTags.length} Tags Restored`}
                  </button>
                )}
                {corruption && (
                  <button className="fvs-btn" id="recover-report-btn" onClick={() => setShowReport(v => !v)}>
                    {showReport ? '▲ Hide Report' : '📊 Corruption Report'}
                  </button>
                )}
                <button className="fvs-btn rw-download-btn" id="recover-dl-strip-btn"
                  onClick={handleDownload} disabled={!result?.recovered_dicom_b64}>
                  ↓ Download DICOM
                </button>
              </div>
            </div>

            {/* ── 3-COLUMN COMPARISON LAYOUT ── */}
            <div className="rw-3col">

              {/* LEFT — Recovery Summary */}
              <aside className="rw-col-left glass-panel">
                <div className="fw-trust-label">RECOVERY SUMMARY</div>
                <div className="rw-psnr-hero">
                  <div className="rw-psnr-value">{affectedPct.toFixed(1)}</div>
                  <div className="rw-psnr-unit">% affected</div>
                  <div className="rw-psnr-desc">Estimated corruption footprint</div>
                </div>
                <div className="rw-status-badge" style={{ color: statusTone }}>
                  <span className="rw-status-dot" aria-hidden="true" />
                  {statusLabel}
                </div>
                <div className="fw-trust-divider" />
                <div className="fw-trust-meta">
                  {[
                    { label: 'Type',       value: corruption?.type     || 'Unknown', color: 'var(--cyan)' },
                    { label: 'Severity',   value: corruption?.severity || 'Unknown', color: 'var(--amber-bright)' },
                    { label: 'Recoverable',value: corruption?.recoverable ? 'Yes' : 'Limited', color: corruption?.recoverable ? 'var(--green-bright)' : 'var(--amber-bright)' },
                    { label: 'Metadata',   value: `${restoredTags.length} restored`,  color: 'var(--green-bright)' },
                  ].map(m => (
                    <div key={m.label} className="fw-tm-row">
                      <span className="fw-tm-label">{m.label}</span>
                      <span className="fw-tm-val" style={{ color: m.color }}>{m.value}</span>
                    </div>
                  ))}
                </div>
                <div className="fw-trust-divider" />
                <button className="rw-big-download-btn" id="recover-big-dl-btn"
                  onClick={handleDownload} disabled={!result?.recovered_dicom_b64}>
                  <span aria-hidden="true">↓</span> Download DICOM
                </button>
                <button className="fw-action-btn ghost rw-reset-btn" id="recover-reset-2-btn" onClick={handleReset}>
                  ↺ New Recovery
                </button>
                <div className="rw-disclaimer">⚡ Approximate reconstruction — not for primary diagnosis.</div>
              </aside>

              {/* CENTER — Wipe Viewer + Compare Pairs below */}
              <main className="rw-col-center">
                <div className="rw-wipe-header">
                  <span className="rw-wipe-title">{singleKey ? `Single View — ${getMeta(singleKey).label}` : 'Interactive Comparison'}</span>
                  <span className="rw-wipe-hint">{singleKey ? 'Click the method again to return to compare' : '← Drag to compare →'}</span>
                </div>
                <WipeViewer imgMap={imgMap} pairIdx={pairIdx} aiAvailable={result.ai_available} singleKey={singleKey} />
                <ComparePairsBar
                  pairIdx={pairIdx} setPairIdx={setPairIdx}
                  singleKey={singleKey} setSingleKey={setSingleKey}
                  aiAvailable={result.ai_available}
                />
              </main>

              {/* RIGHT — Single View selector */}
              <aside className="rw-col-right glass-panel">
                <MethodSelector
                  singleKey={singleKey} setSingleKey={setSingleKey}
                  aiAvailable={result.ai_available}
                />
              </aside>
            </div>


            {/* Metadata expandable */}
            {showMeta && (
              <div className="fw-expandable glass-panel">
                <div className="panel-header">
                  <span className="panel-title">Restored DICOM Metadata</span>
                  <button className="fw-close-btn" id="recover-close-meta" onClick={() => setShowMeta(false)}>✕ Close</button>
                </div>
                <table className="metadata-table rw-meta-table" aria-label="Restored DICOM tags">
                  <thead><tr><th>Tag</th><th>Restored Value</th><th>Status</th></tr></thead>
                  <tbody>
                    {restoredTags.map(([tag, val], i) => (
                      <tr key={i}>
                        <td><span className="tag-name">{tag}</span></td>
                        <td><span className="tag-value">{String(val)}</span></td>
                        <td><span className="tag-status tag-ok">✓ Restored</span></td>
                      </tr>
                    ))}
                    {restoredTags.length === 0 && (
                      <tr><td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '16px' }}>No metadata required restoration</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}

            {/* Corruption report expandable */}
            {showReport && corruption && (
              <div className="fw-expandable glass-panel">
                <div className="panel-header">
                  <span className="panel-title">Corruption Analysis Report</span>
                  <button className="fw-close-btn" id="recover-close-report" onClick={() => setShowReport(false)}>✕ Close</button>
                </div>
                <CorruptionReport report={corruption} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

