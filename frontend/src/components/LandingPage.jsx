import React, { useEffect, useRef, useState } from 'react';

// Animated counter hook
function useCounter(target, duration = 2000) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = Math.ceil(target / (duration / 16));
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setVal(target); clearInterval(timer); }
      else setVal(start);
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);
  return val;
}

export default function LandingPage({ onSelectModule }) {
  const imagesProcessed = useCounter(18472, 2200);
  const threatsDetected = useCounter(3219, 2400);
  const recoveries = useCounter(841, 1800);

  return (
    <main className="landing">

      {/* Hero Section */}
      <section className="landing-hero" aria-labelledby="hero-title">

        {/* AI Powered badge */}
        <div className="hero-badge" aria-label="AI-powered platform">
          <span className="hero-badge-dot" aria-hidden="true" />
          ✦ AI-Powered Medical Image Security
        </div>

        {/* Main title */}
        <div style={{ position: 'relative' }}>
          <h1 className="hero-title" id="hero-title">
            <span className="line1">PhantomaShield</span>
            <span className="line2">Medical Image Integrity Platform</span>
          </h1>
          <div className="hero-title-glow" aria-hidden="true" />
        </div>

        <p className="hero-desc">
          Detect tampered, AI-generated, and corrupted DICOM medical images with
          clinical-grade forensics. Validate metadata integrity and recover damaged scans
          — all in one unified platform.
        </p>

        {/* Stats */}
        <div className="hero-stats" role="region" aria-label="Platform statistics">
          <div className="stat-item">
            <span className="stat-value">99.2%</span>
            <span className="stat-label">Detection Accuracy</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">&lt;&nbsp;3s</span>
            <span className="stat-label">Analysis Time</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">DICOM</span>
            <span className="stat-label">Standard Compliant</span>
          </div>
        </div>

        {/* Live counters */}
        <div className="hero-counters" aria-label="Live system metrics" role="region">
          <div className="counter-item">
            <span className="counter-dot green" aria-hidden="true" />
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{imagesProcessed.toLocaleString()}</span>
            &nbsp;Images Processed
          </div>
          <div className="counter-item">
            <span className="counter-dot amber" aria-hidden="true" />
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{threatsDetected.toLocaleString()}</span>
            &nbsp;Threats Detected
          </div>
          <div className="counter-item">
            <span className="counter-dot cyan" aria-hidden="true" />
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{recoveries.toLocaleString()}</span>
            &nbsp;Files Recovered
          </div>
        </div>
      </section>

      {/* Module Cards */}
      <div className="module-cards" role="navigation" aria-label="Module selection">

        {/* Module 1 — Validate & Detect */}
        <article
          className="module-card detect"
          onClick={() => onSelectModule('detect')}
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && onSelectModule('detect')}
          aria-label="Open DICOM Validation and Detection module"
        >
          <span className="module-card-number">Module 01</span>
          <div className="module-card-icon" aria-hidden="true">🔍</div>
          <h2 className="module-card-title">Validate &amp; Detect</h2>
          <p className="module-card-desc">
            Upload any DICOM file to detect authenticity, localize tampering with
            AI-generated heatmaps, and validate metadata integrity with a trust score.
          </p>
          <ul className="module-card-features" aria-label="Module features">
            {[
              'Authenticity Detection — Real / Tampered / AI-Generated',
              'Grad-CAM Tamper Localization Heatmap',
              'DICOM Metadata Validation & Tag Checking',
              'Trust Score Engine (0–100%)',
            ].map(f => (
              <li key={f}>
                <span className="feature-dot" aria-hidden="true" />
                {f}
              </li>
            ))}
          </ul>
          <button
            className="module-card-cta"
            id="btn-module-detect"
            onClick={e => { e.stopPropagation(); onSelectModule('detect'); }}
          >
            Launch Validation Module
            <span className="cta-arrow" aria-hidden="true">→</span>
          </button>
        </article>

        {/* Module 2 — Corrupted File Recovery */}
        <article
          className="module-card recover"
          onClick={() => onSelectModule('recover')}
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && onSelectModule('recover')}
          aria-label="Open DICOM Corrupted File Recovery module"
        >
          <span className="module-card-number">Module 02</span>
          <div className="module-card-icon" aria-hidden="true">🔧</div>
          <h2 className="module-card-title">Corrupted File Recovery</h2>
          <p className="module-card-desc">
            Detect corruption type and severity in damaged DICOM files, then reconstruct
            missing regions with an AI autoencoder and restore lost metadata tags.
          </p>
          <ul className="module-card-features" aria-label="Module features">
            {[
              'Corruption Type & Severity Analysis',
              'Autoencoder Image Region Reconstruction',
              'OpenCV Inpainting for Damaged Pixels',
              'DICOM Metadata Tag Restoration',
            ].map(f => (
              <li key={f}>
                <span className="feature-dot" aria-hidden="true" />
                {f}
              </li>
            ))}
          </ul>
          <button
            className="module-card-cta"
            id="btn-module-recover"
            onClick={e => { e.stopPropagation(); onSelectModule('recover'); }}
          >
            Launch Recovery Module
            <span className="cta-arrow" aria-hidden="true">→</span>
          </button>
        </article>

      </div>
    </main>
  );
}
