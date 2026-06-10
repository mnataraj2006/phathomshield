import React from 'react';

const SEVERITY_LEVELS = { LOW: 1, MODERATE: 2, HIGH: 3, CRITICAL: 4 };

function SeverityBar({ level }) {
  const n = SEVERITY_LEVELS[level?.toUpperCase()] || 0;
  const color =
    n === 1 ? 'var(--green)'
    : n === 2 ? 'var(--amber)'
    : n === 3 ? 'var(--red)'
    : '#ff2d44';
  const bgColor =
    n === 1 ? 'rgba(16,185,129,0.1)'
    : n === 2 ? 'rgba(245,158,11,0.1)'
    : 'rgba(239,68,68,0.1)';

  return (
    <div className="severity-indicator">
      <div className="severity-dots">
        {[1, 2, 3, 4].map(i => (
          <div
            key={i}
            className="severity-dot"
            style={{
              background: i <= n ? color : 'var(--border)',
              boxShadow: i <= n ? `0 0 8px ${color}` : 'none',
              transition: `all 0.4s ease ${i * 0.08}s`,
            }}
          />
        ))}
      </div>
      <span
        style={{
          fontSize: '0.82rem',
          fontWeight: 700,
          color,
          background: bgColor,
          padding: '2px 10px',
          borderRadius: '999px',
          border: `1px solid ${color}40`,
        }}
      >
        {level || 'Unknown'}
      </span>
    </div>
  );
}

export default function CorruptionReport({ report }) {
  if (!report) return null;

  const {
    type = 'Unknown',
    severity = 'Unknown',
    affected_percentage = 0,
    recoverable = true,
    metadata_issues = 0,
    metadata_restored = 0,
    description = 'Corruption analysis complete.',
  } = report;

  const sevLevel = SEVERITY_LEVELS[severity?.toUpperCase()] || 0;
  const isRecoverable = recoverable !== false;

  return (
    <div className="corruption-report" id="corruption-report-section">
      <div
        className="section-header"
        style={{ padding: 0, paddingBottom: '1.25rem', borderBottom: '1px solid var(--border)', marginBottom: 0 }}
      >
        <span className="section-title">📊 Corruption Analysis Report</span>
        <span
          className={`section-tag ${!isRecoverable ? 'flagged' : sevLevel >= 3 ? 'suspicious' : 'valid'}`}
        >
          {isRecoverable ? '✓ Recoverable' : '✗ Non-recoverable'}
        </span>
      </div>

      <p
        style={{
          fontSize: '0.875rem',
          color: 'var(--text-secondary)',
          marginTop: '1.25rem',
          marginBottom: '1.25rem',
          lineHeight: 1.7,
          padding: '0.875rem 1rem',
          background: 'rgba(255,255,255,0.02)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)',
          borderLeft: '3px solid var(--purple-light)',
        }}
      >
        {description}
      </p>

      <div className="corruption-grid">
        <div className="corruption-item">
          <div className="corruption-item-label">Corruption Type</div>
          <div
            className="corruption-item-value"
            style={{
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.9rem',
            }}
          >
            {type}
          </div>
        </div>

        <div className="corruption-item">
          <div className="corruption-item-label">Severity Level</div>
          <SeverityBar level={severity} />
        </div>

        <div className="corruption-item">
          <div className="corruption-item-label">Affected Area</div>
          <div className="corruption-item-value">
            <span
              style={{
                color: sevLevel >= 3 ? 'var(--red-bright)' : sevLevel >= 2 ? 'var(--amber-bright)' : 'var(--green-bright)',
                fontFamily: 'var(--font-mono)',
                fontSize: '1.1rem',
              }}
            >
              {affected_percentage?.toFixed?.(1) ?? affected_percentage}%
            </span>
          </div>
        </div>

        <div className="corruption-item">
          <div className="corruption-item-label">Metadata Issues Found</div>
          <div
            className="corruption-item-value"
            style={{ color: metadata_issues > 0 ? 'var(--amber-bright)' : 'var(--green-bright)' }}
          >
            {metadata_issues} tags
          </div>
        </div>

        <div className="corruption-item">
          <div className="corruption-item-label">Metadata Restored</div>
          <div className="corruption-item-value" style={{ color: 'var(--green-bright)' }}>
            ✓ {metadata_restored} tags
          </div>
        </div>

        <div className="corruption-item">
          <div className="corruption-item-label">Recovery Method</div>
          <div
            className="corruption-item-value"
            style={{
              fontSize: '0.8rem',
              color: 'var(--purple-light)',
              lineHeight: 1.4,
            }}
          >
            Autoencoder + OpenCV Inpainting
          </div>
        </div>
      </div>
    </div>
  );
}
