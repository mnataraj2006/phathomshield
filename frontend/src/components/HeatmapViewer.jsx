import React, { useEffect, useRef, useState } from 'react';

export default function HeatmapViewer({ original, heatmap, label }) {
  const [heatmapVisible, setHeatmapVisible] = useState(false);

  useEffect(() => {
    if (heatmap || original) {
      // Delay the heatmap appearance slightly for dramatic effect
      const t = setTimeout(() => setHeatmapVisible(true), 400);
      return () => clearTimeout(t);
    }
  }, [heatmap, original]);

  return (
    <div className="results-grid">
      {/* Original Image */}
      <div className="image-card">
        <div className="image-card-header">
          <span>🩻</span>
          Original DICOM Scan
        </div>
        <div className="image-card-body">
          {original ? (
            <img
              src={`data:image/png;base64,${original}`}
              alt="Original DICOM scan"
              id="original-dicom-image"
            />
          ) : (
            <div className="image-placeholder">
              <span style={{ fontSize: '2.5rem', opacity: 0.4 }}>🩻</span>
              <span>No preview available</span>
            </div>
          )}
        </div>
      </div>

      {/* Heatmap / Analysis Overlay */}
      <div
        className="image-card"
        style={{
          borderColor: heatmap ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16,185,129,0.2)',
          transition: 'border-color 0.5s ease',
        }}
      >
        <div className="image-card-header">
          <span>🌡️</span>
          Grad-CAM Forensic Heatmap
          {heatmap && (
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '0.65rem',
                padding: '2px 8px',
                borderRadius: '999px',
                background: 'rgba(245,158,11,0.1)',
                color: 'var(--amber-bright)',
                border: '1px solid rgba(245,158,11,0.25)',
                fontWeight: 700,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                animation: 'livePulse 2s infinite',
              }}
            >
              ⚠ Suspicious
            </span>
          )}
          {!heatmap && original && (
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '0.65rem',
                padding: '2px 8px',
                borderRadius: '999px',
                background: 'rgba(16,185,129,0.1)',
                color: 'var(--green-bright)',
                border: '1px solid rgba(16,185,129,0.25)',
                fontWeight: 700,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}
            >
              ✓ Clean
            </span>
          )}
        </div>
        <div className="image-card-body" style={{ position: 'relative' }}>
          {heatmap ? (
            <>
              <img
                src={`data:image/png;base64,${heatmap}`}
                alt="Grad-CAM tamper localization heatmap overlay"
                id="heatmap-dicom-image"
                style={{
                  opacity: heatmapVisible ? 1 : 0,
                  transition: 'opacity 0.8s ease',
                  animation: heatmapVisible ? 'imgScanIn 0.6s ease both' : 'none',
                }}
              />
              <div
                className="heatmap-overlay-badge"
                data-tooltip="Red areas indicate potential manipulation regions"
              >
                ⚠ Suspicious Region Highlighted
              </div>
            </>
          ) : original ? (
            <>
              <img
                src={`data:image/png;base64,${original}`}
                alt="Original scan – no tampering detected"
                id="heatmap-clean-image"
                style={{
                  filter: 'hue-rotate(120deg) saturate(0.5) brightness(0.85)',
                  opacity: heatmapVisible ? 1 : 0,
                  transition: 'opacity 0.8s ease',
                }}
              />
              <div
                className="heatmap-overlay-badge"
                style={{ borderColor: 'var(--green)', color: 'var(--green-bright)' }}
              >
                ✓ No Suspicious Region Detected
              </div>
            </>
          ) : (
            <div className="image-placeholder">
              <span style={{ fontSize: '2.5rem', opacity: 0.4 }}>🌡️</span>
              <span>Heatmap not generated</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
