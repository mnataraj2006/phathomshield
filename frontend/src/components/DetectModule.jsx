import React, { useState, useCallback } from 'react';
import UploadZone from './UploadZone';
import HeatmapViewer from './HeatmapViewer';
import MetadataTable from './MetadataTable';
import TrustScoreGauge from './TrustScoreGauge';
import { generatePdfReport } from '../utils/generatePdfReport';
import RadiologyReportModule from './RadiologyReportModule';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const DETECT_STEPS = [
  { id: 'upload',     label: 'Initializing forensic engine',             icon: '⚙️' },
  { id: 'preprocess', label: 'Preprocessing image data (OpenCV)',         icon: '🖼️' },
  { id: 'classify',   label: 'Running FFT & noise signal analysis',       icon: '📡' },
  { id: 'heatmap',    label: 'Detecting AI artifacts & textures',         icon: '🤖' },
  { id: 'metadata',   label: 'Validating DICOM metadata integrity',       icon: '🏷️' },
  { id: 'score',      label: 'Computing ensemble trust score',            icon: '🛡️' },
];

function labelBadgeClass(label) {
  if (!label) return '';
  const l = label.toUpperCase();
  if (l.includes('ORIGINAL') || l.includes('REAL')) return 'original';
  if (l.includes('TAMPERED')) return 'tampered';
  return 'ai-generated';
}

function confFillClass(confidence) {
  if (confidence >= 70) return 'green';
  if (confidence >= 40) return 'amber';
  return 'red';
}

function metaStatusClass(status) {
  if (!status) return '';
  const s = status.toUpperCase();
  if (s.includes('VALID')) return 'valid';
  if (s.includes('SUSPICIOUS')) return 'suspicious';
  return 'modified';
}

function buildAIExplanation(result) {
  if (!result) return null;
  const label = (result.label || '').toUpperCase();
  const isOriginal = label.includes('ORIGINAL') || label.includes('REAL');
  const isTampered = label.includes('TAMPERED');
  const f = result.forensics;

  let verdictText = '';
  if (isOriginal) {
    verdictText = 'The forensic analysis indicates this image exhibits natural statistical properties consistent with authentic medical scanner output. FFT frequency distribution follows natural 1/f slope, noise residuals match expected scanner patterns, and texture uniformity is within acceptable bounds.';
  } else if (isTampered) {
    verdictText = 'The forensic analysis detected statistical anomalies consistent with post-acquisition manipulation. Localized pixel discontinuities and inconsistent noise patterns in targeted regions suggest selective pixel-level editing or region splicing.';
  } else {
    verdictText = 'The forensic analysis detected strong synthetic generation signatures. FFT frequency distribution, noise residuals, and DCT block uniformity all fall outside expected natural scanner ranges — consistent with GAN or diffusion model synthesis.';
  }

  return {
    verdictText,
    signals: [
      {
        icon: '📡',
        name: 'FFT Analysis',
        val: f ? `${(f.fft_ai_score * 100).toFixed(0)}%` : '—',
        color: f?.fft_ai_score > 0.45 ? 'var(--red-bright)' : f?.fft_ai_score > 0.25 ? 'var(--amber-bright)' : 'var(--green-bright)',
        desc: 'Frequency domain GAN signature',
      },
      {
        icon: '🔊',
        name: 'Noise Analysis',
        val: f ? `${(f.noise_ai_score * 100).toFixed(0)}%` : '—',
        color: f?.noise_ai_score > 0.45 ? 'var(--red-bright)' : f?.noise_ai_score > 0.25 ? 'var(--amber-bright)' : 'var(--green-bright)',
        desc: 'Noise residual synthetic score',
      },
      {
        icon: '🧩',
        name: 'Texture / DCT',
        val: f ? `${(f.texture_ai_score * 100).toFixed(0)}%` : '—',
        color: f?.texture_ai_score > 0.45 ? 'var(--red-bright)' : f?.texture_ai_score > 0.25 ? 'var(--amber-bright)' : 'var(--green-bright)',
        desc: 'DCT block uniformity ratio',
      },
    ],
  };
}

export default function DetectModule({ onBack }) {
  const [file, setFile] = useState(null);
  const [step, setStep] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setStep(0);

    const stepTimers = DETECT_STEPS.map((_, i) =>
      setTimeout(() => setStep(i), i * 700)
    );

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/validate`, {
        method: 'POST',
        body: formData,
      });
      stepTimers.forEach(clearTimeout);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Server error');
      }

      const data = await res.json();
      setStep(DETECT_STEPS.length);
      setResult(data);
    } catch (e) {
      stepTimers.forEach(clearTimeout);
      setStep(-1);
      setError(e.message || 'Analysis failed. Please check the backend is running.');
    } finally {
      setLoading(false);
    }
  }, [file]);

  const handleReset = () => {
    setResult(null);
    setError(null);
    setStep(-1);
    setFile(null);
  };

  const confidence = result?.confidence ?? 0;
  const trust = result?.trust_score ?? 0;
  const detLabel = result?.label || '—';
  const metaStatus = result?.metadata?.integrity?.status || '—';
  const aiExp = buildAIExplanation(result);
  const progressPct = loading && step >= 0 ? ((step + 1) / DETECT_STEPS.length) * 100 : 0;

  return (
    <main className="module-view">

      {/* Header */}
      <header className="module-header">
        <button className="back-btn" id="detect-back-btn" onClick={onBack} aria-label="Back to module selection">
          ← Back
        </button>
        <div className="module-title-group">
          <h1 style={{ color: 'var(--cyan)' }}>🔍 DICOM Validation &amp; Detection</h1>
          <p>Module 1 — Authenticity analysis, tamper localization &amp; metadata validation</p>
        </div>
      </header>

      {/* Upload Section */}
      {!result && (
        <section className="upload-section" aria-label="File upload">
          <UploadZone
            id="detect-upload-zone"
            onFileSelect={setFile}
            accept=".dcm,.DCM"
            label="DICOM file for analysis"
          />

          {error && (
            <div className="error-banner" role="alert" aria-live="polite">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <button
            className="analyze-btn detect"
            id="detect-analyze-btn"
            onClick={handleAnalyze}
            disabled={!file || loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Analyzing…
              </>
            ) : (
              <>🔍 Launch Forensic Analysis</>
            )}
          </button>
        </section>
      )}

      {/* Progress Steps — AI Pipeline */}
      {loading && (
        <section className="progress-steps" aria-label="Analysis progress" aria-live="polite">
          <p className="progress-steps-title">Forensic AI Pipeline</p>
          <div className="steps-list">
            {DETECT_STEPS.map((s, i) => {
              const state = i < step ? 'done' : i === step ? 'active' : 'pending';
              return (
                <div
                  key={s.id}
                  className={`step-item ${state === 'done' ? 'done-item' : state === 'active' ? 'active-item' : ''}`}
                >
                  <div className={`step-dot ${state}`} aria-hidden="true">
                    {state === 'done' ? '✓' : state === 'active' ? s.icon : i + 1}
                  </div>
                  <span className={`step-label ${state}`}>{s.label}</span>
                </div>
              );
            })}
          </div>
          {/* Progress bar */}
          <div className="step-progress-bar" style={{ marginTop: '1.5rem' }}>
            <div
              className="step-progress-fill"
              style={{ width: `${progressPct}%` }}
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
              role="progressbar"
            />
          </div>
        </section>
      )}

      {/* Results */}
      {result && !loading && (
        <section className="results-section" aria-label="Analysis results">

          {/* Heatmap Images */}
          <HeatmapViewer
            original={result.original_image}
            heatmap={result.heatmap_image}
            label={result.label}
          />

          {/* Metrics Row */}
          <div className="metrics-row" role="region" aria-label="Detection metrics">

            {/* Forensic Detection */}
            <div className="metric-card">
              <div className="metric-card-label">
                <span className="metric-icon">🔬</span>
                Forensic Detection
              </div>
              <div className="metric-status">
                <span className={`status-badge ${labelBadgeClass(detLabel)}`}>
                  {detLabel.includes('ORIGINAL') || detLabel.includes('REAL') ? '✅' :
                   detLabel.includes('TAMPERED') ? '⚠️' : '🤖'}&nbsp;
                  {detLabel}
                </span>
              </div>
              <div className="confidence-bar-wrapper">
                <div className="confidence-label">
                  <span>Model Confidence</span>
                  <span>{confidence.toFixed(1)}%</span>
                </div>
                <div className="confidence-bar" aria-label={`Confidence: ${confidence.toFixed(1)}%`}>
                  <div
                    className={`confidence-fill ${confFillClass(confidence)}`}
                    style={{ width: `${confidence}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Metadata Integrity */}
            <div className="metric-card">
              <div className="metric-card-label">
                <span className="metric-icon">🏷️</span>
                Metadata Integrity
              </div>
              <div className="metric-status">
                <span className={`status-badge ${metaStatusClass(metaStatus)}`}>
                  {metaStatus === 'VALID' ? '✅' : metaStatus === 'SUSPICIOUS' ? '⚠️' : '❌'}&nbsp;
                  {metaStatus}
                </span>
              </div>
              <div className="confidence-bar-wrapper" style={{ marginTop: '0.75rem' }}>
                <div className="confidence-label">
                  <span>Tags Present</span>
                  <span>{result.metadata?.present_count ?? '—'} / {result.metadata?.total_count ?? '—'}</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill cyan"
                    style={{
                      width: result.metadata?.total_count
                        ? `${(result.metadata.present_count / result.metadata.total_count) * 100}%`
                        : '0%'
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Trust Score */}
            <div className="metric-card">
              <div className="metric-card-label">
                <span className="metric-icon">🛡️</span>
                Trust Score Engine
              </div>
              <TrustScoreGauge score={trust} />
            </div>
          </div>

          {/* AI Explanation Panel */}
          {aiExp && (
            <div className="ai-explanation-panel" id="ai-explanation-panel">
              <div className="ai-exp-header">
                <span className="ai-exp-title">
                  🤖 AI Forensic Explanation
                </span>
                <span className="ai-exp-badge">Explainable AI</span>
              </div>
              <div className="ai-exp-body">
                <div className="ai-verdict-block">
                  <div className="ai-verdict-label">Why Image Was Flagged</div>
                  <p className="ai-verdict-text">{aiExp.verdictText}</p>
                </div>
                <div style={{ marginBottom: '0.75rem' }}>
                  <div className="ai-verdict-label" style={{ marginBottom: '0.75rem' }}>
                    Contributing Signal Breakdown
                  </div>
                  <div className="ai-signals-grid">
                    {aiExp.signals.map(sig => (
                      <div key={sig.name} className="ai-signal-item">
                        <span className="ai-signal-icon">{sig.icon}</span>
                        <div className="ai-signal-name">{sig.name}</div>
                        <div className="ai-signal-val" style={{ color: sig.color }}>{sig.val}</div>
                        <div className="ai-signal-desc">{sig.desc}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Forensic Signals Panel */}
          {result.forensics && (
            <div className="forensic-panel" id="forensic-signals-section">
              <div className="section-header">
                <span className="section-title">📊 Forensic Signal Analysis</span>
                <span className={`section-tag ${result.forensics.ai_composite > 0.45 ? 'flagged' : result.forensics.ai_composite > 0.25 ? 'suspicious' : 'valid'}`}>
                  AI Risk: {(result.forensics.ai_composite * 100).toFixed(0)}%
                </span>
              </div>
              <div className="forensic-grid">

                {/* FFT Signal */}
                <div className="signal-block">
                  <div className="signal-label">📡 FFT Frequency Analysis</div>
                  <div className="confidence-bar-wrapper">
                    <div className="confidence-label">
                      <span>GAN Frequency Score</span>
                      <span>{(result.forensics.fft_ai_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill ${result.forensics.fft_ai_score > 0.45 ? 'red' : result.forensics.fft_ai_score > 0.25 ? 'amber' : 'green'}`}
                        style={{ width: `${result.forensics.fft_ai_score * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="signal-meta">
                    Spectral slope:&nbsp;
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                      {result.forensics.spectral_slope?.toFixed(3)}
                    </span>
                    &nbsp;
                    <span style={{ color: (result.forensics.spectral_slope > -0.5 || result.forensics.spectral_slope < -3) ? 'var(--red-bright)' : 'var(--green-bright)' }}>
                      {result.forensics.spectral_slope > -0.5 ? '⚠ Too flat (AI signature)' :
                       result.forensics.spectral_slope < -3 ? '⚠ Too steep (synthetic smoothing)' : '✓ Natural 1/f profile'}
                    </span>
                  </div>
                </div>

                {/* Noise Signal */}
                <div className="signal-block">
                  <div className="signal-label">🔊 Noise Residual Analysis</div>
                  <div className="confidence-bar-wrapper">
                    <div className="confidence-label">
                      <span>Synthetic Noise Score</span>
                      <span>{(result.forensics.noise_ai_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill ${result.forensics.noise_ai_score > 0.45 ? 'red' : result.forensics.noise_ai_score > 0.25 ? 'amber' : 'green'}`}
                        style={{ width: `${result.forensics.noise_ai_score * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="signal-meta">
                    Noise σ:&nbsp;
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                      {result.forensics.noise_std?.toFixed(5)}
                    </span>
                    &nbsp;
                    <span style={{ color: result.forensics.noise_std < 0.003 ? 'var(--red-bright)' : 'var(--green-bright)' }}>
                      {result.forensics.noise_std < 0.003 ? '⚠ Too smooth (synthesized)' : '✓ Natural scanner noise'}
                    </span>
                  </div>
                </div>

                {/* Texture Signal */}
                <div className="signal-block">
                  <div className="signal-label">🧩 Texture / DCT Uniformity</div>
                  <div className="confidence-bar-wrapper">
                    <div className="confidence-label">
                      <span>AI Texture Score</span>
                      <span>{(result.forensics.texture_ai_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill ${result.forensics.texture_ai_score > 0.45 ? 'red' : result.forensics.texture_ai_score > 0.25 ? 'amber' : 'green'}`}
                        style={{ width: `${result.forensics.texture_ai_score * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="signal-meta">
                    DCT blocks:&nbsp;
                    <span style={{ color: result.forensics.dct_uniform ? 'var(--red-bright)' : 'var(--green-bright)', fontWeight: 600 }}>
                      {result.forensics.dct_uniform ? '⚠ Over-uniform (AI signature)' : '✓ Natural DCT diversity'}
                    </span>
                  </div>
                </div>

                {/* Ensemble Verdict */}
                <div className="signal-block">
                  <div className="signal-label">⚖️ Ensemble Verdict</div>
                  <div className="confidence-bar-wrapper">
                    <div className="confidence-label">
                      <span>Combined AI Risk</span>
                      <span style={{
                        color: result.forensics.ai_composite > 0.45 ? 'var(--red-bright)' :
                               result.forensics.ai_composite > 0.25 ? 'var(--amber-bright)' : 'var(--green-bright)',
                        fontWeight: 700
                      }}>
                        {(result.forensics.ai_composite * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill ${result.forensics.ai_composite > 0.45 ? 'red' : result.forensics.ai_composite > 0.25 ? 'amber' : 'green'}`}
                        style={{ width: `${result.forensics.ai_composite * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="confidence-bar-wrapper" style={{ marginTop: 10 }}>
                    <div className="confidence-label">
                      <span>Tamper Risk</span>
                      <span>{(result.forensics.tamper_composite * 100).toFixed(0)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill ${result.forensics.tamper_composite > 0.45 ? 'red' : result.forensics.tamper_composite > 0.25 ? 'amber' : 'green'}`}
                        style={{ width: `${result.forensics.tamper_composite * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* Metadata Table */}
          <MetadataTable
            metadata={result.metadata?.tags}
            integrity={result.metadata?.integrity}
          />

          {/* Radiology Report Generator */}
          <RadiologyReportModule file={file} validationResult={result} />

          {/* Action Buttons */}
          <div className="download-section">
            <button
              className="download-btn primary"
              id="detect-download-report-btn"
              disabled={pdfLoading}
              aria-busy={pdfLoading}
              onClick={async () => {
                setPdfLoading(true);
                try {
                  await generatePdfReport(result, `phantomashield_report_${Date.now()}.pdf`);
                } catch (e) {
                  console.error('PDF generation failed:', e);
                } finally {
                  setPdfLoading(false);
                }
              }}
            >
              {pdfLoading ? (
                <><span className="spinner" aria-hidden="true" /> Generating PDF…</>
              ) : (
                <>📄 Download PDF Report</>
              )}
            </button>
            <button
              className="download-btn secondary"
              id="detect-download-json-btn"
              onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `phantomashield_data_${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              ⬇️ Raw JSON Data
            </button>
            <button
              className="download-btn secondary"
              id="detect-analyze-another-btn"
              onClick={handleReset}
            >
              🔄 Analyze Another
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
