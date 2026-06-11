import React, { useState, useCallback } from 'react';
import UploadZone from './UploadZone';
import RecoveryViewer from './RecoveryViewer';
import CorruptionReport from './CorruptionReport';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const RECOVER_STEPS = [
  { id: 'load',        label: 'Initializing recovery engine',          icon: '⚙️' },
  { id: 'analyze',     label: 'Detecting corruption type & severity',  icon: '🔎' },
  { id: 'reconstruct', label: 'Reconstructing image (Autoencoder)',    icon: '🤖' },
  { id: 'inpaint',     label: 'Filling damaged regions (CV Inpainting)', icon: '🖌️' },
  { id: 'metadata',    label: 'Restoring missing DICOM metadata tags', icon: '🏷️' },
  { id: 'package',     label: 'Packaging recovered DICOM file',        icon: '📦' },
];

export default function RecoverModule({ onBack }) {
  const [file, setFile] = useState(null);
  const [step, setStep] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRecover = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setStep(0);

    const stepTimers = RECOVER_STEPS.map((_, i) =>
      setTimeout(() => setStep(i), i * 900)
    );

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/recover`, {
        method: 'POST',
        body: formData,
      });
      stepTimers.forEach(clearTimeout);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Recovery failed');
      }

      const data = await res.json();
      setStep(RECOVER_STEPS.length);
      setResult(data);
    } catch (e) {
      stepTimers.forEach(clearTimeout);
      setStep(-1);
      setError(e.message || 'Recovery failed. Please check the backend is running.');
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

  const handleDownload = () => {
    if (!result?.recovered_dicom_b64) return;
    const byteChars = atob(result.recovered_dicom_b64);
    const byteNums = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
    const byteArr = new Uint8Array(byteNums);
    const blob = new Blob([byteArr], { type: 'application/dicom' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recovered_${file?.name || 'dicom.dcm'}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const progressPct = loading && step >= 0 ? ((step + 1) / RECOVER_STEPS.length) * 100 : 0;

  return (
    <main className="module-view">

      {/* Header */}
      <header className="module-header">
        <button className="back-btn" id="recover-back-btn" onClick={onBack} aria-label="Back to module selection">
          ← Back
        </button>
        <div className="module-title-group">
          <h1 style={{ color: 'var(--purple-light)' }}>🔧 DICOM Corrupted File Recovery</h1>
          <p>Module 2 — AI-powered reconstruction with Autoencoder + OpenCV Inpainting</p>
        </div>
      </header>

      {/* Upload */}
      {!result && (
        <section className="upload-section" aria-label="File upload for recovery">
          <UploadZone
            id="recover-upload-zone"
            onFileSelect={setFile}
            accept=".dcm,.DCM"
            label="Corrupted DICOM file"
          />

          {error && (
            <div className="error-banner" role="alert" aria-live="polite">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <button
            className="analyze-btn recover"
            id="recover-start-btn"
            onClick={handleRecover}
            disabled={!file || loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Recovering…
              </>
            ) : (
              <>🔧 Start AI Recovery</>
            )}
          </button>
        </section>
      )}

      {/* Recovery Pipeline Progress */}
      {loading && (
        <section
          className="progress-steps"
          aria-label="Recovery progress"
          aria-live="polite"
          style={{ '--accent-color': 'var(--purple-light)', '--accent-glow': 'var(--purple-glow)' }}
        >
          <p className="progress-steps-title">Recovery AI Pipeline</p>
          <div className="steps-list">
            {RECOVER_STEPS.map((s, i) => {
              const state = i < step ? 'done' : i === step ? 'active' : 'pending';
              return (
                <div
                  key={s.id}
                  className={`step-item ${state === 'done' ? 'done-item' : state === 'active' ? 'active-item' : ''}`}
                >
                  <div
                    className={`step-dot ${state}`}
                    style={state === 'active' ? {
                      borderColor: 'var(--purple-light)',
                      color: 'var(--purple-light)',
                      background: 'rgba(167,139,250,0.12)',
                      boxShadow: '0 0 20px rgba(167,139,250,0.5), 0 0 40px rgba(167,139,250,0.2)',
                    } : {}}
                    aria-hidden="true"
                  >
                    {state === 'done' ? '✓' : state === 'active' ? s.icon : i + 1}
                  </div>
                  <span className={`step-label ${state}`}>{s.label}</span>
                </div>
              );
            })}
          </div>
          <div className="step-progress-bar" style={{ marginTop: '1.5rem' }}>
            <div
              className="step-progress-fill"
              style={{
                width: `${progressPct}%`,
                background: 'linear-gradient(90deg, var(--purple), var(--purple-bright))',
                boxShadow: '0 0 12px rgba(167,139,250,0.5)',
              }}
              role="progressbar"
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </section>
      )}

      {/* Results */}
      {result && !loading && (
        <section className="recovery-results" aria-label="Recovery results">

          <RecoveryViewer
            corrupted={result.corrupted_image}
            recovered={result.recovered_image}
          />

          <CorruptionReport report={result.corruption_report} />

          {/* Restored Metadata */}
          {result.restored_metadata && Object.keys(result.restored_metadata).length > 0 && (
            <div className="metadata-section" id="restored-metadata-section">
              <div className="section-header">
                <span className="section-title">🏷️ Restored Metadata Tags</span>
                <span className="section-tag valid">
                  {Object.keys(result.restored_metadata).length} restored
                </span>
              </div>
              <table className="metadata-table" aria-label="Restored DICOM metadata">
                <thead>
                  <tr>
                    <th>Tag</th>
                    <th>Restored Value</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.restored_metadata).map(([tag, val], i) => (
                    <tr key={i}>
                      <td><span className="tag-name">{tag}</span></td>
                      <td><span className="tag-value">{String(val)}</span></td>
                      <td><span className="tag-status tag-ok">✓ Restored</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Disclaimer */}
          <div
            style={{
              background: 'rgba(167, 139, 250, 0.05)',
              border: '1px solid rgba(167,139,250,0.18)',
              borderRadius: 'var(--radius-md)',
              borderLeft: '3px solid var(--purple-light)',
              padding: '1.1rem 1.25rem',
              fontSize: '0.82rem',
              color: 'var(--text-muted)',
              display: 'flex',
              gap: '12px',
              alignItems: 'flex-start',
            }}
          >
            <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>⚡</span>
            <span>
              <strong style={{ color: 'var(--purple-light)', fontWeight: 700 }}>
                Approximate Reconstruction Disclaimer:
              </strong>{' '}
              The recovered image is an AI-assisted approximate reconstruction. It should
              never replace the original scan for clinical diagnosis. Always verify with
              the original source when available.
            </span>
          </div>

          {/* Action Buttons */}
          <div className="download-section">
            <button
              className="download-btn primary"
              id="recover-download-btn"
              onClick={handleDownload}
              disabled={!result?.recovered_dicom_b64}
              style={{
                background: 'linear-gradient(135deg, var(--purple) 0%, var(--purple-bright) 100%)',
                color: '#fff',
                boxShadow: '0 4px 20px rgba(124,58,237,0.3)',
              }}
            >
              ⬇️ Download Recovered DICOM
            </button>
            <button
              className="download-btn secondary"
              id="recover-try-another-btn"
              onClick={handleReset}
            >
              🔄 Try Another File
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
