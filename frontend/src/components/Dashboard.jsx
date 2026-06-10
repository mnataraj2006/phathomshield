import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../utils/api';

function useCounter(target, duration = 1800) {
  const [val, setVal] = useState(0);
  const animated = useRef(false); // only animate once (0 → first real value)

  useEffect(() => {
    if (target === 0) return;
    if (animated.current) {
      // After the initial animation, jump to new values immediately
      setVal(target);
      return;
    }
    animated.current = true;
    let start = 0;
    const step = Math.max(1, Math.ceil(target / (duration / 16)));
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setVal(target); clearInterval(timer); }
      else setVal(start);
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);

  return val;

}

const PIPELINE_STEPS = [
  { icon: '⚙️', label: 'Forensic Engine', desc: 'ResNet-50 loaded · CUDA ready', color: 'var(--cyan)' },
  { icon: '📡', label: 'FFT Analyzer', desc: 'Frequency domain scanner active', color: 'var(--purple-light)' },
  { icon: '🤖', label: 'AI Detector', desc: 'GAN/Diffusion signature model', color: 'var(--cyan)' },
  { icon: '🛡️', label: 'Trust Engine', desc: 'Ensemble scoring · v2.1', color: 'var(--green-bright)' },
  { icon: '🔧', label: 'Recovery Engine', desc: 'U-Net autoencoder · PSNR 24.6dB', color: 'var(--purple-light)' },
];

const BACKEND = API_BASE;

const ACTION_COLOR = {
  ORIGINAL:      'var(--green-bright)',
  TAMPERED:      'var(--red-bright)',
  'AI-GENERATED':'var(--amber-bright)',
  RECOVERED:     'var(--purple-light)',
  REPORT:        'var(--cyan)',
};

const STATUS_META = {
  Safe:          { color: 'var(--green-bright)',  bg: 'rgba(16,185,129,0.1)',  border: 'rgba(16,185,129,0.25)',  icon: '✅' },
  Tampered:      { color: 'var(--red-bright)',    bg: 'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.25)',   icon: '⚠️' },
  'AI-Generated':{ color: 'var(--amber-bright)', bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.25)', icon: '🤖' },
};

// ─── Activity hook ─────────────────────────────────────────────────────────────
function useActivity(pollMs = 10000) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const fetchActivity = async () => {
      try {
        const res = await fetch(`${BACKEND}/activity?limit=5`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setItems(data.activity || []);
      } catch (_) {
        // backend offline — leave list as-is
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchActivity();
    const timer = setInterval(fetchActivity, pollMs);
    return () => { cancelled = true; clearInterval(timer); };
  }, [pollMs]);

  return { items, loading };
}

// ─── Stats hook ───────────────────────────────────────────────────────────────
function useStats(pollMs = 20000) {
  const [stats, setStats] = useState({
    scans_processed:    0,
    threats_detected:   0,
    files_recovered:    0,
    detection_accuracy: 80.4,
  });

  useEffect(() => {
    let cancelled = false;
    const fetchStats = async () => {
      try {
        const res = await fetch(`${BACKEND}/stats`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setStats(data);
      } catch (_) {
        // backend offline — keep previous values
      }
    };
    fetchStats();
    const timer = setInterval(fetchStats, pollMs);
    return () => { cancelled = true; clearInterval(timer); };
  }, [pollMs]);

  return stats;
}

// ─── Cases hook ────────────────────────────────────────────────────────────────
function useCases(pollMs = 15000) {
  const [cases, setCases] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const fetchCases = async () => {
      try {
        const res = await fetch(`${BACKEND}/cases?limit=50`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          setCases(data.cases || []);
          setTotal(data.total || 0);
        }
      } catch (_) {
        // backend offline
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchCases();
    const timer = setInterval(fetchCases, pollMs);
    return () => { cancelled = true; clearInterval(timer); };
  }, [pollMs]);

  return { cases, total, loading };
}

// ─── Case Card ─────────────────────────────────────────────────────────────────
function CaseCard({ c }) {
  const sm = STATUS_META[c.status] || { color: 'var(--text-muted)', bg: 'rgba(255,255,255,0.04)', border: 'rgba(255,255,255,0.1)', icon: '🩻' };
  const imgSrc = c.image_path ? `${BACKEND}/images/${c.image_path}` : null;

  const dateLabel = c.date
    ? new Date(c.date).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
    : '—';

  return (
    <article className="case-card glass-panel" id={`case-${c.id}`} aria-label={`Case ${c.file_name}`}>
      {/* Image preview */}
      <div className="case-card-image">
        {imgSrc
          ? <img src={imgSrc} alt={`Preview of ${c.file_name}`} loading="lazy" />
          : <div className="case-card-image-placeholder" aria-hidden="true">🩻</div>
        }
        <div className="case-card-badges">
          {c.has_validation && <span className="case-badge validate-badge" title="Validation">V</span>}
          {c.has_recovery   && <span className="case-badge recover-badge"  title="Recovery">R</span>}
        </div>
      </div>

      {/* Body */}
      <div className="case-card-body">
        <div className="case-card-filename" title={c.file_name}>{c.file_name}</div>

        {c.status && (
          <div
            className="case-card-status"
            style={{ color: sm.color, background: sm.bg, borderColor: sm.border }}
          >
            <span>{sm.icon}</span>
            <span>{c.status}</span>
          </div>
        )}

        <div className="case-card-metrics">
          {c.trust_score != null && (
            <div className="case-metric">
              <span className="case-metric-label">Trust</span>
              <span className="case-metric-value" style={{
                color: c.trust_score >= 70 ? 'var(--green-bright)' : c.trust_score >= 40 ? 'var(--amber-bright)' : 'var(--red-bright)',
              }}>
                {c.trust_score}%
              </span>
            </div>
          )}
          {c.quality_score != null && (
            <div className="case-metric">
              <span className="case-metric-label">Quality</span>
              <span className="case-metric-value" style={{ color: 'var(--purple-light)' }}>
                {(c.quality_score * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {c.recovered != null && (
            <div className="case-metric">
              <span className="case-metric-label">Recovered</span>
              <span className="case-metric-value" style={{ color: c.recovered ? 'var(--green-bright)' : 'var(--red-bright)' }}>
                {c.recovered ? '✓' : '✗'}
              </span>
            </div>
          )}
        </div>

        <div className="case-card-footer">
          <span className="case-card-date">📅 {dateLabel}</span>
          {c.severity && (
            <span className="case-card-severity" style={{
              color: c.severity === 'critical' ? 'var(--red-bright)'
                   : c.severity === 'high'     ? 'var(--amber-bright)'
                   : 'var(--text-muted)',
            }}>
              {c.severity}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

// ─── Dashboard ─────────────────────────────────────────────────────────────────
export default function Dashboard({ onNavigate }) {
  const { items: activity, loading: activityLoading } = useActivity();
  const { cases, total, loading: casesLoading } = useCases();
  const stats = useStats();

  const totalScans = useCounter(stats.scans_processed, 1800);
  const threats    = useCounter(stats.threats_detected, 2000);
  const recoveries = useCounter(stats.files_recovered, 1600);
  const [activePulse, setActivePulse] = useState(0);
  const [caseFilter, setCaseFilter] = useState('all');

  useEffect(() => {
    const t = setInterval(() => setActivePulse(p => (p + 1) % PIPELINE_STEPS.length), 1400);
    return () => clearInterval(t);
  }, []);

  const filteredCases = cases.filter(c => {
    if (caseFilter === 'validation') return c.has_validation;
    if (caseFilter === 'recovery')   return c.has_recovery;
    return true;
  });

  return (
    <div className="dashboard" id="dashboard-view">
      {/* Hero Header */}
      <section className="dash-hero" aria-labelledby="dash-hero-title">
        <div className="dash-hero-badge" aria-label="AI Security Platform">
          <span className="dash-hero-badge-dot" aria-hidden="true" />
          Clinical-Grade AI Forensics
        </div>
        <h1 className="dash-hero-title" id="dash-hero-title">
          <span className="dash-title-main">PhantomaShield</span>
          <span className="dash-title-sub">Medical Image Integrity Platform</span>
        </h1>
        <p className="dash-hero-desc">
          Enterprise-grade DICOM forensics powered by deep learning. Detect tampered, AI-generated,
          and corrupted medical images with sub-3-second clinical accuracy.
        </p>

        <div className="dash-cta-row">
          <button className="dash-cta-btn primary" id="dash-cta-detect" onClick={() => onNavigate('detect')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            Launch Forensic Detection
          </button>
          <button className="dash-cta-btn secondary" id="dash-cta-recover" onClick={() => onNavigate('recover')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
            Start AI Recovery
          </button>
        </div>
      </section>

      {/* Metrics Row */}
      <section className="dash-metrics" aria-label="Platform statistics">
        {[
          {
            value:    totalScans.toLocaleString(),
            label:    'Scans Processed',
            sub:      'Total DICOM files analysed',
            icon:     '🩻',
            color:    'var(--cyan)',
            glow:     'rgba(0,245,255,0.18)',
            border:   'rgba(0,245,255,0.3)',
            bg:       'rgba(0,245,255,0.05)',
            bars:     [40, 65, 50, 80, 70, 90, 60],
            progress: Math.min(100, (stats.scans_processed / 20) * 100),
            trend:    '+12%',
          },
          {
            value:    threats.toLocaleString(),
            label:    'Threats Detected',
            sub:      'Tampered · AI-generated',
            icon:     '⚠️',
            color:    'var(--red-bright)',
            glow:     'rgba(239,68,68,0.18)',
            border:   'rgba(239,68,68,0.3)',
            bg:       'rgba(239,68,68,0.05)',
            bars:     [30, 55, 40, 70, 45, 80, 50],
            progress: stats.scans_processed > 0 ? Math.min(100, (stats.threats_detected / stats.scans_processed) * 100) : 0,
            trend:    'Active',
          },
          {
            value:    recoveries.toLocaleString(),
            label:    'Files Recovered',
            sub:      'AI reconstruction success',
            icon:     '✨',
            color:    'var(--purple-light)',
            glow:     'rgba(167,139,250,0.18)',
            border:   'rgba(167,139,250,0.3)',
            bg:       'rgba(167,139,250,0.05)',
            bars:     [20, 45, 60, 35, 75, 55, 80],
            progress: Math.min(100, (stats.files_recovered / 10) * 100),
            trend:    '+8%',
          },
          {
            value:    `${stats.detection_accuracy}%`,
            label:    'Detection Accuracy',
            sub:      'ResNet-50 ensemble model',
            icon:     '🎯',
            color:    'var(--green-bright)',
            glow:     'rgba(16,185,129,0.18)',
            border:   'rgba(16,185,129,0.3)',
            bg:       'rgba(16,185,129,0.05)',
            bars:     [70, 78, 75, 82, 79, 80, 80],
            progress: stats.detection_accuracy,
            trend:    'Stable',
          },
        ].map((m, i) => (
          <div
            key={m.label}
            className="dash-metric-card"
            style={{ '--card-color': m.color, '--card-glow': m.glow, '--card-border': m.border, '--card-bg': m.bg, animationDelay: `${i * 0.08}s` }}
          >
            {/* Ambient glow blob */}
            <div className="dmc-glow-blob" aria-hidden="true" />

            {/* Shimmer top border */}
            <div className="dmc-top-border" aria-hidden="true" />

            {/* Header row: icon + trend chip */}
            <div className="dmc-header">
              <div className="dmc-icon-wrap" aria-hidden="true">{m.icon}</div>
              <span className="dmc-trend-chip">{m.trend}</span>
            </div>

            {/* Main value */}
            <div className="dmc-value" style={{ color: m.color }}>{m.value}</div>

            {/* Label + sub */}
            <div className="dmc-label">{m.label}</div>
            <div className="dmc-sub">{m.sub}</div>

            {/* Mini sparkline */}
            <div className="dmc-sparkline" aria-hidden="true">
              {m.bars.map((h, bi) => (
                <div
                  key={bi}
                  className="dmc-spark-bar"
                  style={{ height: `${h}%`, background: m.color, opacity: bi === m.bars.length - 1 ? 1 : 0.35 + bi * 0.08 }}
                />
              ))}
            </div>

            {/* Progress bar */}
            <div className="dmc-progress-track" aria-hidden="true">
              <div
                className="dmc-progress-fill"
                style={{ width: `${Math.max(2, m.progress)}%`, background: `linear-gradient(90deg, ${m.color}88, ${m.color})` }}
              />
            </div>
          </div>
        ))}
      </section>


      {/* Two-Column Lower */}
      <div className="dash-lower">
        {/* AI Pipeline Status */}
        <section className="dash-pipeline glass-panel" aria-label="AI Pipeline status">
          <div className="panel-header">
            <span className="panel-title">AI Analysis Pipeline</span>
            <span className="panel-badge live">RUNNING</span>
          </div>
          <div className="pipeline-steps" role="list">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={step.label} className={`pipeline-step${i === activePulse ? ' active' : ''}`} role="listitem">
                <div className="pipeline-step-icon" style={{ color: step.color }} aria-hidden="true">{step.icon}</div>
                <div className="pipeline-step-content">
                  <div className="pipeline-step-name">{step.label}</div>
                  <div className="pipeline-step-desc">{step.desc}</div>
                </div>
                <div className="pipeline-step-status" aria-hidden="true">
                  <span className="pipeline-dot" style={{ background: step.color }} />
                </div>
              </div>
            ))}
          </div>
          <div className="pipeline-progress-track" aria-hidden="true">
            <div className="pipeline-progress-fill" />
          </div>
        </section>

        {/* Recent Activity */}
        <section className="dash-activity glass-panel" aria-label="Recent analysis activity">
          <div className="panel-header">
            <span className="panel-title">Recent Activity</span>
            <span className="panel-badge">Live Feed</span>
          </div>
          <div className="activity-list" role="list">
            {activityLoading && <div className="activity-empty">Loading…</div>}
            {!activityLoading && activity.length === 0 && (
              <div className="activity-empty">
                <span style={{ fontSize: '1.5rem' }}>📭</span>
                <span>No scans yet — upload a DICOM to get started.</span>
              </div>
            )}
            {activity.map((item, i) => {
              const color = ACTION_COLOR[item.action] || 'var(--cyan)';
              const ts = new Date(item.timestamp);
              const diffMs = Date.now() - ts.getTime();
              const diffMin = Math.round(diffMs / 60000);
              const timeLabel = diffMin < 1 ? 'just now'
                : diffMin === 1 ? '1m ago'
                : diffMin < 60 ? `${diffMin}m ago`
                : `${Math.round(diffMin / 60)}h ago`;
              return (
                <div key={item.result_id || i} className="activity-item" role="listitem" style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="activity-dot" style={{ background: color }} aria-hidden="true" />
                  <div className="activity-content">
                    <span className="activity-name">{item.filename}</span>
                    <span className="activity-time">{timeLabel}</span>
                  </div>
                  <span className="activity-action" style={{ color }}>{item.action}</span>
                </div>
              );
            })}
          </div>
          <div className="capability-cards">
            <div className="capability-card detect-card" onClick={() => onNavigate('detect')} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && onNavigate('detect')} id="dash-capability-detect">
              <div className="capability-card-title">Forensic Detection</div>
              <div className="capability-card-desc">Real · Tampered · AI-Gen</div>
              <div className="capability-card-arrow">→</div>
            </div>
            <div className="capability-card recover-card" onClick={() => onNavigate('recover')} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && onNavigate('recover')} id="dash-capability-recover">
              <div className="capability-card-title">AI Recovery</div>
              <div className="capability-card-desc">Autoencoder · Inpainting</div>
              <div className="capability-card-arrow">→</div>
            </div>
          </div>
        </section>
      </div>

      {/* ── Cases Archive Section ── */}
      <section className="dash-cases-section" aria-labelledby="cases-heading">
        <div className="cases-section-header">
          <div className="cases-section-title-row">
            <h2 className="cases-section-title" id="cases-heading">
              Case Archive
              {total > 0 && <span className="cases-count-badge">{total}</span>}
            </h2>
            <p className="cases-section-sub">All processed DICOM cases — persisted locally on disk</p>
          </div>
          <div className="cases-filter-tabs" role="tablist" aria-label="Filter cases by module">
            {[
              { key: 'all',        label: 'All Cases' },
              { key: 'validation', label: 'Validation' },
              { key: 'recovery',   label: 'Recovery' },
            ].map(tab => (
              <button
                key={tab.key}
                role="tab"
                aria-selected={caseFilter === tab.key}
                className={`cases-tab${caseFilter === tab.key ? ' active' : ''}`}
                id={`cases-tab-${tab.key}`}
                onClick={() => setCaseFilter(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {casesLoading && (
          <div className="cases-empty">
            <span className="cases-empty-icon">⏳</span>
            <span>Loading cases…</span>
          </div>
        )}
        {!casesLoading && filteredCases.length === 0 && (
          <div className="cases-empty">
            <span className="cases-empty-icon">📂</span>
            <span>No cases found. Run a forensic detection or recovery to populate this archive.</span>
          </div>
        )}
        {!casesLoading && filteredCases.length > 0 && (
          <div className="cases-grid" role="list">
            {filteredCases.map((c, i) => (
              <div key={c.id} role="listitem" style={{ animationDelay: `${i * 0.05}s` }}>
                <CaseCard c={c} />
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
