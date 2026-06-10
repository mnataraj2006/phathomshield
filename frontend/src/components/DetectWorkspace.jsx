import React, { useState, useCallback } from 'react';
import UploadZone from './UploadZone';
import HeatmapViewer from './HeatmapViewer';
import MetadataTable from './MetadataTable';
import { generatePdfReport } from '../utils/generatePdfReport';
import RadiologyReportModule from './RadiologyReportModule';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PIPELINE_STEPS = [
  { id: 'init',      icon: '⚙️', label: 'Initializing Model',    desc: 'ResNet-50 · GPU ready' },
  { id: 'fft',       icon: '📡', label: 'FFT Frequency Scan',     desc: 'Spectral analysis' },
  { id: 'artifacts', icon: '🔬', label: 'Detecting AI Artifacts', desc: 'GAN signatures' },
  { id: 'metadata',  icon: '🏷️', label: 'Validating Metadata',   desc: 'DICOM tag integrity' },
  { id: 'trust',     icon: '🛡️', label: 'Computing Trust Score', desc: 'Ensemble verdict' },
];

function getTrustMeta(score) {
  if (score >= 85) return {
    label: 'SAFE', color: '#34d399', ring: '#34d399',
    rec: 'Forensic signals consistent with authentic scanner output. Safe threshold reached.',
  };
  if (score >= 60) return {
    label: 'REVIEW REQUIRED', color: '#fbbf24', ring: '#fbbf24',
    rec: 'Borderline anomaly detected. Expert validation required before any clinical use.',
  };
  return {
    label: 'REJECT', color: '#f87171', ring: '#f87171',
    rec: 'Multiple high-severity forensic anomalies. Do not use for clinical decisions.',
  };
}

function labelBadgeClass(label) {
  if (!label) return '';
  const l = label.toUpperCase();
  if (l.includes('ORIGINAL') || l.includes('REAL')) return 'original';
  if (l.includes('TAMPERED')) return 'tampered';
  return 'ai-generated';
}

function buildReasoningItems(result) {
  if (!result?.forensics) return [];
  const f = result.forensics;
  const metadataStatus = result.metadata?.integrity?.status;
  return [
    {
      ok: f.fft_ai_score <= 0.45,
      title: f.fft_ai_score > 0.45 ? 'Abnormal frequency pattern detected' : 'Frequency distribution matches real scanners',
      detail: `FFT spectral slope of ${f.spectral_slope?.toFixed(2)} — ${f.fft_ai_score > 0.45 ? 'deviation from natural 1/f pattern suggests synthetic origin' : 'consistent with natural medical imaging hardware'}`,
    },
    {
      ok: f.noise_ai_score <= 0.45,
      title: f.noise_ai_score > 0.45 ? 'Noise residual is unnaturally smooth' : 'Scanner noise pattern verified',
      detail: `Noise σ = ${f.noise_std?.toFixed(5)} — ${f.noise_ai_score > 0.45 ? 'too clean for real acquisition hardware' : 'matches expected sensor noise profile'}`,
    },
    {
      ok: f.texture_ai_score <= 0.45,
      title: f.texture_ai_score > 0.45 ? 'GAN artifact signature in texture' : 'Texture diversity is natural',
      detail: `DCT block uniformity ${f.dct_uniform ? 'detected — over-uniform blocks are a hallmark of generative models' : 'not detected — natural variation is present across regions'}`,
    },
    {
      ok: metadataStatus === 'VALID',
      title: metadataStatus === 'VALID' ? 'Metadata consistency verified' : 'Metadata anomalies found',
      detail: metadataStatus === 'VALID'
        ? 'All DICOM tags pass integrity checks — no modifications detected'
        : 'One or more DICOM tags show signs of modification or inconsistency',
    },
  ];
}

function buildSignalCards(result) {
  if (!result?.forensics) return [];
  const f = result.forensics;
  const risk = s => s > 0.45 ? 'high' : s > 0.25 ? 'medium' : 'low';
  const label = (s, low, med, high) => s > 0.45 ? high : s > 0.25 ? med : low;
  return [
    { icon: '📡', name: 'Frequency (FFT)', score: f.fft_ai_score,     risk: risk(f.fft_ai_score),     label: label(f.fft_ai_score, 'Natural spectrum', 'Slight deviation', 'Abnormal spectral pattern') },
    { icon: '🔊', name: 'Noise Residual',  score: f.noise_ai_score,   risk: risk(f.noise_ai_score),   label: label(f.noise_ai_score, 'Natural noise', 'Minor anomaly', 'Unnatural noise level') },
    { icon: '🧩', name: 'Texture / DCT',   score: f.texture_ai_score, risk: risk(f.texture_ai_score), label: label(f.texture_ai_score, 'Natural diversity', 'Minor uniformity', 'GAN artifact signature') },
    { icon: '⚖️', name: 'Tamper Risk',     score: f.tamper_composite, risk: risk(f.tamper_composite), label: label(f.tamper_composite, 'Low risk', 'Moderate risk', 'High manipulation risk') },
  ];
}

function TrustRing({ score, meta }) {
  const radius = 54;
  const circ = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, score));
  const dash = (pct / 100) * circ;
  return (
    <div className="trust-ring-wrap">
      <svg width="148" height="148" viewBox="0 0 148 148" aria-label={`Trust score: ${Math.round(pct)} out of 100`}>
        <circle cx="74" cy="74" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10"/>
        <circle cx="74" cy="74" r={radius} fill="none" stroke={meta.ring} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          strokeDashoffset={circ / 4}
          style={{ filter: `drop-shadow(0 0 8px ${meta.color})`, transition: 'stroke-dasharray 1.4s cubic-bezier(0.4,0,0.2,1)' }}
        />
        <text x="74" y="70" textAnchor="middle" fill="white" fontSize="28" fontWeight="800"
          fontFamily="Space Grotesk, Inter, sans-serif" style={{ letterSpacing: '-0.03em' }}>
          {Math.round(pct)}
        </text>
        <text x="74" y="86" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="11" fontFamily="Inter, sans-serif">
          / 100
        </text>
      </svg>
    </div>
  );
}

export default function DetectWorkspace() {
  const [file, setFile]             = useState(null);
  const [step, setStep]             = useState(-1);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [showMeta, setShowMeta]     = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [heatMode, setHeatMode]     = useState('original'); // 'original' | 'heatmap' | 'overlay'
  const [fullscreenImg, setFullscreenImg] = useState(null); // base64 string | null

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true); setResult(null); setError(null); setStep(0);
    setShowMeta(false); setShowReport(false);

    const timers = PIPELINE_STEPS.map((_, i) => setTimeout(() => setStep(i), i * 800));
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/validate`, { method: 'POST', body: fd });
      timers.forEach(clearTimeout);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Server error');
      }
      const data = await res.json();
      setStep(PIPELINE_STEPS.length);
      setResult(data);
    } catch (e) {
      timers.forEach(clearTimeout);
      setStep(-1);
      setError(e.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleReset = () => {
    setResult(null); setError(null); setStep(-1); setFile(null);
    setShowMeta(false); setShowReport(false);
  };

  const trust     = result?.trust_score ?? 0;
  const trustMeta = getTrustMeta(trust);
  const detLabel  = result?.label || '—';
  const confidence = result?.confidence ?? 0;
  const progressPct = loading && step >= 0 ? ((step + 1) / PIPELINE_STEPS.length) * 100 : 0;
  const reasoning  = result?.forensic_reasoning || buildReasoningItems(result);
  const signals    = buildSignalCards(result);
  const fusion     = result?.fusion || null;
  const tier       = result?.confidence_tier || 'CONFIRMED';
  const clinRel    = result?.clinical_reliability || 'LIKELY_AUTHENTIC';
  const conclusion = result?.conclusion || '';

  return (
    <div className="fw-workspace" id="detect-workspace">

      {/* ─── TOP BAR ─────────────────────────────────── */}
      <div className="fw-topbar">
        <div className="fw-topbar-left">
          <div className="fw-topbar-icon detect-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <div>
            <h1 className="fw-topbar-title">Forensic Detection</h1>
            <p className="fw-topbar-sub">DICOM authenticity · Tamper localization · Trust scoring</p>
          </div>
        </div>
        <div className="fw-topbar-right">
          <div className="fw-status-pill">
            <span className="fw-status-pulse" aria-hidden="true"/>
            <span>AI Engine Active</span>
          </div>
          {result && (
            <>
              <button className="fw-action-btn ghost" id="detect-new-btn" onClick={handleReset}>↺ New Analysis</button>
              <button className="fw-action-btn primary" id="detect-pdf-btn" disabled={pdfLoading}
                onClick={async () => {
                  setPdfLoading(true);
                  try { await generatePdfReport(result, `phantomashield_${Date.now()}.pdf`); }
                  catch (e) { console.error(e); }
                  finally { setPdfLoading(false); }
                }}>
                {pdfLoading ? <><span className="btn-spinner"/>Exporting…</> : '↓ Export Report'}
              </button>
            </>
          )}
        </div>
      </div>

      {/* ─── SCROLLABLE CONTENT ──────────────────────── */}
      <div className="fw-content">

        {/* ══ IDLE ══════════════════════════════════════ */}
        {!result && !loading && (
          <div className="fw-idle">

            {/* Upload Hero */}
            <div className="fw-upload-hero-block">
              <div className="fw-step-label">STEP 1 — UPLOAD FILE</div>
              <UploadZone
                id="detect-upload-zone"
                onFileSelect={setFile}
                accept=".dcm,.DCM"
                label="Upload DICOM file"
              />
              {error && (
                <div className="fw-error-bar" role="alert" aria-live="polite">
                  <span aria-hidden="true">⚠</span> {error}
                </div>
              )}
              <button
                className={`fw-cta-btn${!file ? ' fw-cta-disabled' : ''}`}
                id="detect-analyze-btn"
                onClick={handleAnalyze}
                disabled={!file}
                aria-label="Start AI forensic analysis"
              >
                <span className="fw-cta-glow" aria-hidden="true"/>
                <span className="fw-cta-icon" aria-hidden="true">⚡</span>
                <span className="fw-cta-label">Start AI Analysis</span>
                <span className="fw-cta-hint">{file ? file.name : 'Upload a .dcm file to begin'}</span>
              </button>
            </div>

            {/* Pipeline Preview */}
            <div className="fw-pipeline-preview">
              <div className="fw-step-label">ANALYSIS PIPELINE — 5 STEPS</div>
              <div className="fw-steps-row">
                {PIPELINE_STEPS.map((s, i) => (
                  <div key={s.id} className="fw-step-card fw-step-idle">
                    <span className="fw-step-num-badge">{i + 1}</span>
                    <span className="fw-step-icon">{s.icon}</span>
                    <span className="fw-step-name">{s.label}</span>
                    <span className="fw-step-desc">{s.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══ PROCESSING ════════════════════════════════ */}
        {loading && (
          <div className="fw-processing">
            <div className="fw-proc-header">
              <div className="fw-proc-orbs" aria-hidden="true">
                <div className="fw-orb fw-orb-a"/><div className="fw-orb fw-orb-b"/>
              </div>
              <div className="fw-proc-title">AI Engine Processing</div>
              <div className="fw-proc-file" aria-live="polite">{file?.name}</div>
            </div>

            <div className="fw-steps-row fw-steps-active" role="list" aria-label="Analysis pipeline">
              {PIPELINE_STEPS.map((s, i) => {
                const state = i < step ? 'done' : i === step ? 'active' : 'pending';
                return (
                  <div key={s.id} className={`fw-step-card fw-step-${state}`} role="listitem">
                    <div className="fw-step-indicator" aria-hidden="true">
                      {state === 'done'   && <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>}
                      {state === 'active' && <span className="fw-step-spinner"/>}
                      {state === 'pending'&& <span className="fw-step-num">{i + 1}</span>}
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
                <div className="fw-progress-fill" style={{ width: `${progressPct}%` }} role="progressbar" aria-valuenow={progressPct} aria-valuemin={0} aria-valuemax={100}/>
              </div>
              <div className="fw-progress-label" aria-live="polite">{PIPELINE_STEPS[step]?.label || 'Initializing…'}</div>
            </div>

            <div className="fw-proc-wave" aria-hidden="true">
              {Array.from({ length: 24 }, (_, i) => (
                <div key={i} className="fw-wave-bar" style={{ animationDelay: `${i * 0.06}s` }}/>
              ))}
            </div>
          </div>
        )}

        {/* ══ RESULTS ═══════════════════════════════════ */}
        {result && !loading && (
          <div className="fw-results">

            {/* Verdict Strip */}
            <div className={`fw-verdict-strip fvs-${labelBadgeClass(detLabel)}`}>
              <div className="fvs-left">
                <span className={`status-badge ${labelBadgeClass(detLabel)}`}>{detLabel}</span>
                <span className="fvs-conf">Model confidence: {confidence.toFixed(1)}%</span>
                {/* Confidence tier badge */}
                {tier === 'HUMAN_REVIEW_REQUIRED' && (
                  <span className="fw-tier-badge tier-human-review">
                    ⚠ Human Review Required
                  </span>
                )}
                {tier === 'UNCERTAIN' && (
                  <span className="fw-tier-badge tier-uncertain">
                    ⚠ Low Confidence — Verify Manually
                  </span>
                )}
              </div>
              <div className="fvs-right">
                <button className="fvs-btn" id="detect-meta-btn" onClick={() => setShowMeta(v => !v)}>
                  {showMeta ? '▲ Hide Metadata' : '▼ View Metadata'}
                </button>
                <button className="fvs-btn" id="detect-report-btn" onClick={() => setShowReport(v => !v)}>
                  {showReport ? '▲ Hide Report' : '📄 AI Report'}
                </button>
              </div>
            </div>

            {/* Confidence warning banner */}
            {tier !== 'CONFIRMED' && (
              <div className={`fw-conf-banner ${tier === 'HUMAN_REVIEW_REQUIRED' ? 'banner-review' : 'banner-uncertain'}`}>
                <span className="fw-banner-icon">⚠️</span>
                <div>
                  <div className="fw-banner-title">
                    {tier === 'HUMAN_REVIEW_REQUIRED'
                      ? 'Human Review Required — Model confidence below safe threshold (60%)'
                      : 'Preliminary Detection — Low Confidence Result (60–69%)'}
                  </div>
                  <div className="fw-banner-sub">
                    Model confidence is below the safe threshold. Output should not be used for automated clinical decisions. Please involve a qualified radiologist.
                  </div>
                </div>
              </div>
            )}

            {/* Clinical reliability banner + disclaimer */}
            <div className={`fw-reliability-banner rel-${clinRel.toLowerCase()}`}>
              <span>{result?.reliability_label}</span>
              <span className="fw-rel-score">Trust Score: {trust}/100</span>
            </div>
            <div className="fw-clinical-disclaimer" role="note">
              ⚠️ <strong>Not for clinical use without verification</strong> — PhantomaShield is a forensic aid tool.
              All results require qualified radiologist review before any diagnostic or clinical workflow use.
            </div>

            {/* ROW 1: Trust Hero + Image Comparison */}
            <div className="fw-row1">

              {/* Trust Score Hero */}
              <div className="fw-trust-card glass-panel">
                <div className="fw-trust-label">TRUST SCORE</div>
                <TrustRing score={trust} meta={trustMeta}/>
                <div className="fw-trust-verdict" style={{ color: trustMeta.color }}>
                  ● {trustMeta.label}
                </div>
                <div className="fw-trust-rec">{trustMeta.rec}</div>
                <div className="fw-trust-divider"/>
                <div className="fw-trust-meta">
                  <div className="fw-tm-row">
                    <span className="fw-tm-label">Composite AI Risk</span>
                    <span className="fw-tm-val" style={{
                      color: result.forensics?.ai_composite > 0.45 ? 'var(--red-bright)'
                           : result.forensics?.ai_composite > 0.25 ? 'var(--amber-bright)'
                           : 'var(--green-bright)'
                    }}>
                      {((result.forensics?.ai_composite || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="fw-tm-row">
                    <span className="fw-tm-label">Tamper Composite</span>
                    <span className="fw-tm-val">{((result.forensics?.tamper_composite || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="fw-tm-row">
                    <span className="fw-tm-label">Model Confidence</span>
                    <span className="fw-tm-val" style={{ color: confidence >= 85 ? 'var(--green-bright)' : confidence >= 60 ? 'var(--amber-bright)' : 'var(--red-bright)' }}>
                      {confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="fw-tm-row">
                    <span className="fw-tm-label">Tier</span>
                    <span className="fw-tm-val" style={{ fontSize: '0.72rem', color: tier === 'CONFIRMED' ? 'var(--green-bright)' : 'var(--amber-bright)' }}>{tier.replace(/_/g, ' ')}</span>
                  </div>
                </div>
              </div>

              {/* Image Comparison with 3-mode toggle */}
              <div className="fw-images glass-panel" style={{ maxWidth: '800px', flex: '1 1 420px' }}>
                <div className="panel-header">
                  <span className="panel-title">Image Analysis</span>
                  {/* 3-mode toggle */}
                  <div className="fw-img-toggle" role="group" aria-label="Image view mode">
                    {[['original','Original'],['heatmap','Heatmap'],['overlay','Overlay']].map(([mode, label]) => (
                      <button
                        key={mode}
                        className={`fw-img-toggle-btn${heatMode===mode ? ' active' : ''}`}
                        id={`detect-view-${mode}`}
                        onClick={() => setHeatMode(mode)}
                        aria-pressed={heatMode===mode}
                      >{label}</button>
                    ))}
                  </div>
                </div>
                {/* Image display */}
                <div className="fw-img-frame">
                  {/* Current image src helper */}
                  {(() => {
                    const src = heatMode === 'original' ? result.original_image
                              : heatMode === 'heatmap'  ? result.heatmap_image
                              : result.overlay_image;
                    const alt = heatMode === 'original' ? 'Original DICOM scan'
                              : heatMode === 'heatmap'  ? 'Grad-CAM forensic heatmap'
                              : 'Contour overlay';
                    return src ? (
                      <>
                        <img
                          src={`data:image/png;base64,${src}`}
                          alt={alt}
                          className="fw-img-display"
                        />
                        {/* Fullscreen zoom button */}
                        <button
                          className="fw-img-zoom-btn"
                          id="detect-img-fullscreen"
                          onClick={() => setFullscreenImg(src)}
                          aria-label="View fullscreen"
                          title="Expand image"
                        >
                          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
                          </svg>
                          Fullscreen
                        </button>
                        {/* Heatmap legend */}
                        {heatMode === 'heatmap' && (
                          <div className="fw-img-legend">
                            <span className="fw-legend-label">
                              {trust < 40 ? 'High Risk Region Identified' 
                               : trust <= 70 ? 'Suspicious Patterns Detected' 
                               : 'No Significant Risk Detected'}
                            </span>
                            <div className="fw-legend-bar">
                              {['#1a237e','#0d47a1','#00bcd4','#4caf50','#ffeb3b','#ff5722','#b71c1c'].map(c => (
                                <span key={c} style={{background:c}}/>
                              ))}
                            </div>
                            <div className="fw-legend-ticks"><span>Low</span><span>High</span></div>
                          </div>
                        )}
                        {heatMode === 'overlay' && (
                          <div className="fw-img-caption">Cyan contours = top 25% activation boundaries</div>
                        )}
                      </>
                    ) : null;
                  })()}
                </div>
              </div>
            </div>

            {/* ROW 2: AI Reasoning + Signal Breakdown */}
            <div className="fw-row2">

              {/* AI Reasoning — severity-rated from backend */}
              <div className="fw-reasoning glass-panel">
                <div className="panel-header">
                  <span className="panel-title">Forensic Evidence</span>
                  <span className="panel-badge success">Severity Rated</span>
                </div>
                <div className="fw-reasoning-list">
                  {(result.forensic_reasoning?.length ? result.forensic_reasoning : reasoning).map((item, i) => {
                    // Support both backend format ({signal,severity,message}) and legacy format ({ok,title,detail})
                    const isBackend = 'signal' in item;
                    const sev = isBackend ? item.severity : (item.ok ? 'LOW' : 'HIGH');
                    const title = isBackend ? item.signal : item.title;
                    const detail = isBackend ? item.message : item.detail;
                    const sevClass = sev === 'HIGH' ? 'ri-high' : sev === 'MEDIUM' ? 'ri-medium' : 'ri-low';
                    const icon = sev === 'HIGH' ? '⚠' : sev === 'MEDIUM' ? '⚬' : '✓';
                    return (
                      <div key={i} className={`fw-ri-item ${sevClass}`}>
                        <div className="fw-ri-left">
                          <span className="fw-ri-dot" aria-hidden="true">{icon}</span>
                          <span className={`fw-ri-sev sev-${sev.toLowerCase()}`}>{sev}</span>
                        </div>
                        <div>
                          <div className="fw-ri-title">{title}</div>
                          <div className="fw-ri-detail">{detail}</div>
                          {isBackend && (
                            <div className="fw-ri-score">Score: {(item.value * 100).toFixed(0)}%</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {/* Conclusion */}
                {conclusion && (
                  <div className="fw-conclusion">
                    <span className="fw-conclusion-label">🧐 Conclusion</span>
                    <p className="fw-conclusion-text">{conclusion}</p>
                  </div>
                )}
              </div>

              {/* Signal Breakdown */}
              <div className="fw-signals glass-panel">
                <div className="panel-header">
                  <span className="panel-title">Signal Breakdown</span>
                  <span className={`panel-badge ${result.forensics?.ai_composite > 0.45 ? 'danger' : result.forensics?.ai_composite > 0.25 ? 'warn' : 'success'}`}>
                    AI Risk {((result.forensics?.ai_composite || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="fw-signal-grid">
                  {signals.map(sig => (
                    <div key={sig.name} className={`fw-sig-card fsc-risk-${sig.risk}`}>
                      <div className="fw-sig-top">
                        <span className="fw-sig-icon" aria-hidden="true">{sig.icon}</span>
                        <span className="fw-sig-name">{sig.name}</span>
                        <span className="fw-sig-pct">{(sig.score * 100).toFixed(0)}%</span>
                      </div>
                      <div className="fw-sig-track">
                        <div className="fw-sig-fill" style={{ width: `${sig.score * 100}%` }}/>
                      </div>
                      <div className="fw-sig-label">{sig.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Fusion Transparency Panel */}
            {fusion && (
              <div className="fw-fusion glass-panel">
                <div className="panel-header">
                  <span className="panel-title">⚡ Multi-Signal Fusion Logic</span>
                  <span className="panel-badge success">Mathematical Transparency</span>
                </div>
                <div className="fw-fusion-desc">
                  Final decision is derived from a weighted multi-signal ensemble fusion. Weights are calibrated based on forensic reliability of each signal type.
                </div>
                <div className="fw-fusion-formula">
                  <code>Trust = 0.40 × CNN + 0.20 × (1−FFT) + 0.20 × (1−Noise) + 0.20 × (1−Texture)</code>
                </div>
                <div className="fw-fusion-bars">
                  {[
                    { label: 'CNN Confidence',    weight: fusion.cnn_weight,     value: fusion.cnn_value,     contrib: fusion.cnn_contribution,     color: '#06b6d4' },
                    { label: 'FFT Anomaly Score', weight: fusion.fft_weight,     value: fusion.fft_value,     contrib: fusion.fft_contribution,     color: '#a855f7' },
                    { label: 'Noise Residual',    weight: fusion.noise_weight,   value: fusion.noise_value,   contrib: fusion.noise_contribution,   color: '#f59e0b' },
                    { label: 'Texture / DCT',     weight: fusion.texture_weight, value: fusion.texture_value, contrib: fusion.texture_contribution, color: '#10b981' },
                    { label: 'Heatmap Evidence',  weight: fusion.heatmap_weight, value: fusion.heatmap_value, contrib: fusion.heatmap_contribution, color: '#ef4444' },
                  ].map(c => (
                    <div key={c.label} className="fw-fusion-bar-row">
                      <div className="fw-fusion-bar-info">
                        <span className="fw-fusion-bar-label">{c.label}</span>
                        <span className="fw-fusion-bar-weight">w={c.weight.toFixed(2)}</span>
                        <span className="fw-fusion-bar-val">v={c.value.toFixed(2)}</span>
                        <span className="fw-fusion-bar-contrib">→ {c.contrib.toFixed(3)}</span>
                      </div>
                      <div className="fw-fusion-track">
                        <div
                          className="fw-fusion-fill"
                          style={{ width: `${c.value * 100}%`, background: c.color }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="fw-fusion-total">
                  Fused Trust Score: <strong>{trust}/100</strong>
                  <span className="fw-fusion-note">Metadata penalty applied separately · Hard caps for high-confidence negative labels</span>
                </div>
              </div>
            )}

            {/* Expandable Metadata */}
            {showMeta && (
              <div className="fw-expandable glass-panel">
                <div className="panel-header">
                  <span className="panel-title">DICOM Metadata</span>
                  <button className="fw-close-btn" id="detect-close-meta" onClick={() => setShowMeta(false)}>✕ Close</button>
                </div>
                <MetadataTable metadata={result.metadata?.tags} integrity={result.metadata?.integrity}/>
              </div>
            )}

            {/* Expandable Radiology Report */}
            {showReport && (
              <div className="fw-expandable glass-panel">
                <div className="panel-header">
                  <span className="panel-title">AI Radiology Report</span>
                  <button className="fw-close-btn" id="detect-close-report" onClick={() => setShowReport(false)}>✕ Close</button>
                </div>
                <RadiologyReportModule file={file} validationResult={result}/>
              </div>
            )}

          </div>
        )}
      </div>

      {/* ─── Fullscreen Image Modal ──────────────── */}
      {fullscreenImg && (
        <div
          className="fw-fullscreen-modal"
          role="dialog"
          aria-modal="true"
          aria-label="Fullscreen image viewer"
          onClick={() => setFullscreenImg(null)}
          onKeyDown={e => e.key === 'Escape' && setFullscreenImg(null)}
          tabIndex={-1}
        >
          <button
            className="fw-fullscreen-close"
            id="detect-fullscreen-close"
            onClick={() => setFullscreenImg(null)}
            aria-label="Close fullscreen"
          >✕ Close</button>
          <img
            src={`data:image/png;base64,${fullscreenImg}`}
            alt="Fullscreen DICOM view"
            onClick={e => e.stopPropagation()}
          />
          <div className="fw-fullscreen-label">Click anywhere to close · ESC to dismiss</div>
        </div>
      )}
    </div>
  );
}
