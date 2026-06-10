import React, { useState } from 'react';
import { generateRadiologyPdf } from '../utils/generateRadiologyPdf';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/* ── Shared sub-components ───────────────────────────────────────────── */
function SectionBox({ number, icon, title, children }) {
  return (
    <div style={{
      marginBottom: '1rem',
      border: '1px solid var(--border)',
      borderRadius: 10,
      overflow: 'hidden',
      background: 'rgba(255,255,255,0.02)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '0.55rem 1.1rem',
        background: 'rgba(0,229,255,0.05)',
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{
          width: 22, height: 22, borderRadius: '50%',
          background: 'rgba(0,229,255,0.15)',
          border: '1px solid rgba(0,229,255,0.4)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '0.65rem', fontWeight: 800, color: 'var(--cyan)', flexShrink: 0,
        }}>{number}</span>
        <span style={{ fontSize: '0.88rem' }}>{icon}</span>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--cyan)' }}>
          {title}
        </span>
      </div>
      <div style={{ padding: '0.9rem 1.2rem' }}>{children}</div>
    </div>
  );
}

function InfoGrid({ items }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.6rem' }}>
      {items.map(([label, value]) => value && value !== 'N/A' && value !== 'None' ? (
        <div key={label} style={{
          background: 'rgba(255,255,255,0.03)',
          borderRadius: 6, padding: '0.5rem 0.75rem',
          border: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{ fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 3 }}>
            {label}
          </div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            {String(value)}
          </div>
        </div>
      ) : null)}
    </div>
  );
}

function Finding({ label, value }) {
  if (!value || value === 'N/A') return null;
  return (
    <div style={{ display: 'flex', gap: '1rem', padding: '0.55rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)', alignItems: 'flex-start' }}>
      <span style={{ minWidth: 150, fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', paddingTop: 2, flexShrink: 0 }}>
        {label}
      </span>
      <span style={{ fontSize: '0.855rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
        {value}
      </span>
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────────────────── */
export default function RadiologyReportModule({ file, validationResult }) {
  const [loading, setLoading]     = useState(false);
  const [report, setReport]       = useState(null);
  const [error, setError]         = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  const [countdown, setCountdown]   = useState(0);

  const startCountdown = (seconds, file) => {
    setCountdown(seconds);
    const iv = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(iv);
          handleGenerate(file, true); // auto-retry
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleGenerate = async (fileArg, isRetry = false) => {
    const targetFile = fileArg || file;
    if (!targetFile) return;
    if (!isRetry) { setReport(null); setError(null); setCountdown(0); }
    setLoading(true); setError(null);
    try {
      const formData = new FormData();
      formData.append('file', targetFile);
      const res = await fetch(`${API_BASE}/radiology-report`, { method: 'POST', body: formData });
      if (res.status === 429) {
        setLoading(false);
        startCountdown(60, targetFile);
        return;
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Server error');
      }
      setReport(await res.json());
    } catch (e) {
      setError(e.message || 'Report generation failed.');
    } finally {
      setLoading(false);
    }
  };

  const handlePdf = async () => {
    if (!report) return;
    setPdfLoading(true);
    try { await generateRadiologyPdf(report, validationResult); }
    catch (e) { console.error('PDF failed:', e); }
    finally { setPdfLoading(false); }
  };

  const r        = report?.report || {};
  const pi       = r.patient_info || {};
  const si       = r.study_info   || {};
  const findings = r.findings     || {};
  const impress  = r.impression   || [];
  const label    = report?.label  || 'UNKNOWN';
  const isAlert  = label !== 'ORIGINAL';
  const alertC   = isAlert ? 'var(--red)' : 'var(--green)';

  return (
    <div id="radiology-report-module" style={{
      marginTop: '2rem',
      border: '1px solid rgba(0,229,255,0.18)',
      borderRadius: 14,
      background: 'rgba(8,11,26,0.75)',
      overflow: 'hidden',
    }}>
      {/* ── Header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '1rem 1.4rem',
        background: 'rgba(0,229,255,0.05)',
        borderBottom: '1px solid rgba(0,229,255,0.12)',
        flexWrap: 'wrap', gap: 10,
      }}>
        <div>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--cyan)' }}>
            🏥 AI Clinical Radiology Report
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
            Standard 7-Section Format · Powered by Gemini Vision AI · Preliminary review only
          </div>
        </div>

        {!report && countdown === 0 && (
          <button id="generate-radiology-report-btn" className="analyze-btn detect"
            style={{ padding: '0.55rem 1.3rem', fontSize: '0.83rem' }}
            onClick={() => handleGenerate()} disabled={loading || !file} aria-busy={loading}>
            {loading ? <><span className="spinner" aria-hidden="true" /> Generating…</> : <>🩺 Generate Report</>}
          </button>
        )}

        {countdown > 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.3)',
            borderRadius: 8, padding: '0.5rem 1rem',
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: '50%',
              border: '3px solid rgba(245,158,11,0.3)',
              borderTop: '3px solid var(--amber)',
              animation: 'spin 1s linear infinite', flexShrink: 0,
            }} />
            <div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--amber)' }}>
                Rate limited — auto-retrying in {countdown}s
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>
                Free tier: 15 requests/minute. Will retry automatically.
              </div>
            </div>
          </div>
        )}

        {report && (
          <button id="download-radiology-pdf-btn" className="download-btn primary"
            style={{ padding: '0.55rem 1.3rem', fontSize: '0.83rem' }}
            onClick={handlePdf} disabled={pdfLoading}>
            {pdfLoading ? <><span className="spinner" aria-hidden="true" /> Building PDF…</> : <>📄 Download Radiology PDF</>}
          </button>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="error-banner" role="alert" style={{ margin: '1rem 1.4rem' }}>
          <span>⚠️</span><span>{error}</span>
        </div>
      )}

      {/* ── Loading ── */}
      {loading && (
        <div style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <div className="spinner" style={{ width: 30, height: 30, margin: '0 auto 1rem', borderWidth: 3 }} />
          <p style={{ fontSize: '0.85rem' }}>Gemini Vision is analyzing the DICOM image…</p>
          <p style={{ fontSize: '0.72rem', marginTop: 6, opacity: 0.6 }}>This may take 10–20 seconds</p>
        </div>
      )}

      {/* ── Report Body ── */}
      {report && r && (
        <div style={{ padding: '1.3rem' }}>

          {/* Forensic Integrity Alert */}
          {isAlert && (
            <div style={{
              display: 'flex', gap: 10, padding: '0.85rem 1.1rem',
              background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.28)',
              borderRadius: 9, marginBottom: '1rem',
            }}>
              <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>⚠️</span>
              <div>
                <div style={{ fontWeight: 700, color: 'var(--red)', fontSize: '0.8rem', marginBottom: 3 }}>
                  AUTHENTICITY ALERT — {label}
                </div>
                <div style={{ fontSize: '0.79rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {r.integrity_note}
                </div>
              </div>
            </div>
          )}

          {/* Report header strip */}
          <div style={{
            display: 'flex', gap: '0.6rem', marginBottom: '1rem', flexWrap: 'wrap',
          }}>
            {[
              { label: 'Report Type',   value: r.report_type, color: 'var(--cyan)'  },
              { label: 'Trust Score',   value: `${Math.round(report.trust_score || 0)}%`, color: report.trust_score >= 70 ? 'var(--green)' : report.trust_score >= 40 ? 'var(--amber)' : 'var(--red)' },
              { label: 'Confidence',    value: `${(report.confidence || 0).toFixed(1)}%`,  color: 'var(--text-primary)' },
              { label: 'Generated By',  value: report.generated_by, color: 'var(--text-muted)' },
            ].map(({ label: lbl, value, color }) => (
              <div key={lbl} style={{
                flex: '1 1 140px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--border)',
                borderRadius: 7, padding: '0.5rem 0.85rem',
              }}>
                <div style={{ fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 3 }}>{lbl}</div>
                <div style={{ fontWeight: 700, fontSize: '0.88rem', color }}>{value || '—'}</div>
              </div>
            ))}
          </div>

          {/* SECTION 1 — Patient Information */}
          <SectionBox number="1" icon="🧾" title="Patient Information">
            <InfoGrid items={[
              ['Patient Name',    pi.patient_name],
              ['Age',            pi.age],
              ['Sex',            pi.sex],
              ['Patient ID/MRN', pi.patient_id],
              ['Referring Doctor', pi.referring_doctor],
              ['Hospital',       pi.hospital],
            ]} />
          </SectionBox>

          {/* SECTION 2 — Study Information */}
          <SectionBox number="2" icon="📅" title="Study Information">
            <InfoGrid items={[
              ['Study Date',      si.study_date],
              ['Study Time',      si.study_time],
              ['Modality',        si.modality],
              ['Body Part',       si.body_part],
              ['Accession No.',   si.accession_number],
            ]} />
          </SectionBox>

          {/* SECTION 3 — Clinical Indication */}
          <SectionBox number="3" icon="⚙️" title="Clinical Indication">
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
              {r.clinical_indication || '—'}
            </p>
          </SectionBox>

          {/* SECTION 4 — Technique */}
          <SectionBox number="4" icon="🛠️" title="Technique / Procedure">
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, margin: 0 }}>
              {r.technique || '—'}
            </p>
          </SectionBox>

          {/* SECTION 5 — Findings */}
          <SectionBox number="5" icon="🔍" title="Findings">
            <Finding label="Image Quality"        value={findings.image_quality} />
            <Finding label="Lungs / Parenchyma"  value={findings.lungs} />
            <Finding label="Pleura"              value={findings.pleura} />
            <Finding label="Mediastinum"         value={findings.mediastinum} />
            <Finding label="Heart & Vessels"     value={findings.heart_vessels} />
            <Finding label="Bones & Soft Tissue" value={findings.bones_soft_tissue} />
            {findings.other && findings.other !== 'N/A' && (
              <Finding label="Other"             value={findings.other} />
            )}
          </SectionBox>

          {/* SECTION 6 — Impression */}
          <SectionBox number="6" icon="🧠" title="Impression (Conclusion)">
            <ol style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {impress.map((imp, i) => (
                <li key={i} style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 6, fontWeight: i === 0 ? 600 : 400 }}>
                  {imp}
                </li>
              ))}
            </ol>
          </SectionBox>

          {/* SECTION 7 — Recommendations */}
          {r.recommendation && (
            <SectionBox number="7" icon="⚠️" title="Recommendations">
              <p style={{ fontSize: '0.875rem', color: 'var(--cyan)', lineHeight: 1.7, margin: 0, fontWeight: 600 }}>
                {r.recommendation}
              </p>
            </SectionBox>
          )}

          {/* Disclaimer */}
          <div style={{
            padding: '0.7rem 1rem', marginTop: '0.5rem',
            background: 'rgba(245,158,11,0.05)',
            border: '1px solid rgba(245,158,11,0.18)',
            borderRadius: 8, fontSize: '0.7rem', color: 'var(--amber)', lineHeight: 1.6,
          }}>
            ⚠ <strong>Disclaimer:</strong> {r.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}
