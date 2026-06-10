import React from 'react';

const FRIENDLY_TAGS = {
  '(0008,0060)': 'Modality',
  '(0020,000D)': 'Study Instance UID',
  '(0020,000E)': 'Series Instance UID',
  '(0008,0020)': 'Study Date',
  '(0008,0030)': 'Study Time',
  '(0010,0010)': 'Patient Name',
  '(0010,0020)': 'Patient ID',
  '(0010,0030)': 'Patient Birth Date',
  '(0010,0040)': 'Patient Sex',
  '(0008,103E)': 'Series Description',
  '(0028,0010)': 'Rows',
  '(0028,0011)': 'Columns',
  '(0028,0030)': 'Pixel Spacing',
  '(0028,0100)': 'Bits Allocated',
  '(0028,0101)': 'Bits Stored',
  '(0028,1050)': 'Window Center',
  '(0028,1051)': 'Window Width',
  '(0008,0016)': 'SOP Class UID',
  '(0008,0018)': 'SOP Instance UID',
  '(0008,0070)': 'Manufacturer',
  '(0008,1090)': 'Manufacturer Model Name',
  '(0018,0050)': 'Slice Thickness',
  '(0020,0013)': 'Instance Number',
};

export default function MetadataTable({ metadata, integrity }) {
  if (!metadata || Object.keys(metadata).length === 0) {
    return (
      <div className="metadata-section">
        <div className="section-header">
          <span className="section-title">🏷️ DICOM Metadata</span>
        </div>
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          No metadata available
        </div>
      </div>
    );
  }

  const integrityTag = integrity?.status === 'VALID' ? 'valid'
    : integrity?.status === 'SUSPICIOUS' ? 'suspicious' : 'flagged';

  const aiIndicators = integrity?.ai_indicators || [];
  const warnings = integrity?.warnings || [];
  const suspicionScore = integrity?.suspicion_score || 0;
  const entries = Object.entries(metadata);

  return (
    <div className="metadata-section" id="metadata-table-section">
      <div className="section-header">
        <span className="section-title">🏷️ DICOM Metadata Tags</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {suspicionScore > 0 && (
            <span style={{
              fontSize: '0.7rem', fontWeight: 700, padding: '3px 10px',
              borderRadius: 'var(--radius-full)',
              background: suspicionScore >= 35 ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
              color: suspicionScore >= 35 ? 'var(--red-bright)' : 'var(--amber-bright)',
              border: `1px solid ${suspicionScore >= 35 ? 'rgba(239,68,68,0.3)' : 'rgba(245,158,11,0.3)'}`,
              textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              Suspicion: {suspicionScore}/100
            </span>
          )}
          <span className={`section-tag ${integrityTag}`}>
            {integrity?.status || 'checked'}
          </span>
        </div>
      </div>

      {/* AI Indicators Banner */}
      {aiIndicators.length > 0 && (
        <div style={{
          margin: '0',
          padding: '1rem 1.5rem',
          background: 'rgba(239,68,68,0.06)',
          borderBottom: '1px solid rgba(239,68,68,0.2)',
        }}>
          <p style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--red)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
            🤖 AI/Synthetic Generation Indicators
          </p>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: 4, listStyle: 'none' }}>
            {aiIndicators.map((ind, i) => (
              <li key={i} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: 8 }}>
                <span style={{ color: 'var(--red)', flexShrink: 0 }}>⚠</span>
                {ind}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div style={{
          padding: '0.75rem 1.5rem',
          background: 'rgba(245,158,11,0.04)',
          borderBottom: '1px solid rgba(245,158,11,0.15)',
        }}>
          <p style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--amber)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.4rem' }}>
            ⚠ Metadata Warnings
          </p>
          <ul style={{ display: 'flex', flexDirection: 'column', gap: 3, listStyle: 'none' }}>
            {warnings.map((w, i) => (
              <li key={i} style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>• {w}</li>
            ))}
          </ul>
        </div>
      )}

      {integrity?.hash && (
        <div style={{ padding: '0.75rem 1.5rem', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          File Hash (MD5): <span style={{ color: 'var(--cyan)' }}>{integrity.hash}</span>
        </div>
      )}

      <table className="metadata-table" aria-label="DICOM metadata tags">
        <thead>
          <tr>
            <th>Tag</th>
            <th>Description</th>
            <th>Value</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([tag, value], i) => {
            const isPresent = value !== null && value !== undefined && value !== '';
            const friendly = FRIENDLY_TAGS[tag] || 'Unknown Tag';
            const statusClass = isPresent ? 'tag-ok' : 'tag-err';
            const statusIcon = isPresent ? '✓' : '✗';
            return (
              <tr key={i}>
                <td><span className="tag-name">{tag}</span></td>
                <td style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{friendly}</td>
                <td><span className="tag-value" title={isPresent ? String(value) : ''}>{isPresent ? String(value) : '—'}</span></td>
                <td>
                  <span className={`tag-status ${statusClass}`}>
                    {statusIcon} {isPresent ? 'Present' : 'Missing'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
