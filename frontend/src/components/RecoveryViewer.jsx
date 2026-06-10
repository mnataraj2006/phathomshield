import React from 'react';

export default function RecoveryViewer({ corrupted, recovered }) {
  return (
    <div className="results-grid">
      {/* Corrupted Image */}
      <div
        className="image-card"
        style={{ borderColor: 'rgba(239,68,68,0.2)' }}
      >
        <div className="image-card-header">
          <span>💔</span>
          Corrupted DICOM Input
          <span
            style={{
              marginLeft: 'auto',
              fontSize: '0.65rem',
              padding: '2px 8px',
              borderRadius: '999px',
              background: 'rgba(239,68,68,0.1)',
              color: 'var(--red-bright)',
              border: '1px solid rgba(239,68,68,0.25)',
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}
          >
            ✗ Corrupted
          </span>
        </div>
        <div className="image-card-body">
          {corrupted ? (
            <img
              src={`data:image/png;base64,${corrupted}`}
              alt="Corrupted DICOM scan input"
              id="corrupted-dicom-image"
              style={{
                filter: 'contrast(1.05) brightness(0.8) saturate(0.6)',
              }}
            />
          ) : (
            <div className="image-placeholder">
              <span style={{ fontSize: '2.5rem', opacity: 0.3, filter: 'grayscale(1)' }}>🩻</span>
              <span>Corrupted image preview</span>
            </div>
          )}
        </div>
      </div>

      {/* Recovered Image */}
      <div
        className="image-card"
        style={{ borderColor: 'rgba(167,139,250,0.25)' }}
      >
        <div className="image-card-header">
          <span>✨</span>
          AI-Recovered DICOM
          <span
            style={{
              marginLeft: 'auto',
              fontSize: '0.65rem',
              padding: '2px 8px',
              borderRadius: '999px',
              background: 'rgba(167,139,250,0.1)',
              color: 'var(--purple-light)',
              border: '1px solid rgba(167,139,250,0.25)',
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}
          >
            ⚡ Reconstructed
          </span>
        </div>
        <div className="image-card-body" style={{ position: 'relative' }}>
          {recovered ? (
            <>
              <img
                src={`data:image/png;base64,${recovered}`}
                alt="Approximate reconstructed DICOM scan"
                id="recovered-dicom-image"
                style={{
                  animation: 'imgScanIn 0.8s ease 0.2s both',
                }}
              />
              <div
                className="approx-badge"
                data-tooltip="AI-assisted approximate reconstruction — not for clinical diagnosis"
              >
                ⚡ Approximate Reconstruction
              </div>
            </>
          ) : (
            <div className="image-placeholder">
              <span style={{ fontSize: '2.5rem', opacity: 0.4 }}>✨</span>
              <span>Recovered image will appear here</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
