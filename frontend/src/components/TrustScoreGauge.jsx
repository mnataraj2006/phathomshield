import React, { useEffect, useRef } from 'react';

function getSemiArcPath(cx, cy, r, startDeg, endDeg) {
  const toRad = d => (d * Math.PI) / 180;
  const sx = cx + r * Math.cos(toRad(startDeg));
  const sy = cy + r * Math.sin(toRad(startDeg));
  const ex = cx + r * Math.cos(toRad(endDeg));
  const ey = cy + r * Math.sin(toRad(endDeg));
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`;
}

export default function TrustScoreGauge({ score }) {
  const s = Math.min(100, Math.max(0, score || 0));

  // Animated color based on score
  const color = s >= 70 ? '#10b981' : s >= 40 ? '#f59e0b' : '#ef4444';
  const glowColor = s >= 70
    ? 'rgba(16,185,129,0.6)'
    : s >= 40 ? 'rgba(245,158,11,0.6)'
    : 'rgba(239,68,68,0.6)';
  const label = s > 70 ? 'No Significant Risk' : s >= 40 ? 'Suspicious Patterns Detected' : 'High Risk Detected';
  const recommendation = s >= 70
    ? '✓ Safe to use for diagnostics'
    : s >= 40 ? '⚠ Manual verification recommended'
    : '✗ Do not use — suspected manipulation';

  // Circular ring approach
  const size = 140;
  const cx = size / 2;
  const cy = size / 2;
  const strokeWidth = 10;
  const r = (size / 2) - strokeWidth - 2;
  const circumference = 2 * Math.PI * r;
  // Show 75% of circumference as the arc (from top, going clockwise 270 deg)
  const arcAngle = 270; // degrees
  const arcFraction = arcAngle / 360;
  const arcLength = circumference * arcFraction;
  const gap = circumference - arcLength;
  // Fill based on score
  const filledArc = arcLength * (s / 100);
  const strokeDasharray = `${filledArc} ${circumference - filledArc}`;
  // Rotate so arc starts at 135deg (bottom-left) and sweeps clockwise to 45deg (bottom-right)
  const rotateAngle = 135; // degrees

  return (
    <div className="trust-score-container" id="trust-score-gauge">
      <div
        className="trust-gauge"
        style={{ width: size, height: size }}
        aria-label={`Trust score: ${s}%`}
      >
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img">
          <defs>
            <linearGradient id="trustGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={s >= 70 ? '#10b981' : s >= 40 ? '#f59e0b' : '#ef4444'} />
              <stop offset="100%" stopColor={s >= 70 ? '#34d399' : s >= 40 ? '#fbbf24' : '#f87171'} />
            </linearGradient>
            <filter id="trustGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Track (background arc) */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth={strokeWidth}
            strokeDasharray={`${arcLength} ${circumference - arcLength}`}
            strokeDashoffset={0}
            strokeLinecap="round"
            transform={`rotate(${rotateAngle} ${cx} ${cy})`}
          />

          {/* Fill arc */}
          {s > 0 && (
            <circle
              cx={cx} cy={cy} r={r}
              fill="none"
              stroke="url(#trustGradient)"
              strokeWidth={strokeWidth}
              strokeDasharray={strokeDasharray}
              strokeDashoffset={0}
              strokeLinecap="round"
              transform={`rotate(${rotateAngle} ${cx} ${cy})`}
              filter="url(#trustGlow)"
              style={{
                transition: 'stroke-dasharray 1.5s cubic-bezier(0.4, 0, 0.2, 1)',
                filter: `drop-shadow(0 0 6px ${glowColor})`
              }}
            />
          )}

          {/* Score value */}
          <text
            x={cx}
            y={cy - 4}
            textAnchor="middle"
            fill={color}
            fontSize="26"
            fontWeight="800"
            fontFamily="'JetBrains Mono', monospace"
            style={{ filter: `drop-shadow(0 0 8px ${glowColor})` }}
          >
            {s}
          </text>
          <text
            x={cx}
            y={cy + 16}
            textAnchor="middle"
            fill={color}
            fontSize="11"
            fontWeight="600"
            fontFamily="'Inter', sans-serif"
            opacity="0.8"
          >
            / 100
          </text>
        </svg>
      </div>

      <div style={{ textAlign: 'center' }}>
        <div
          className="trust-recommendation"
          style={{
            color,
            background: s >= 70
              ? 'rgba(16,185,129,0.1)'
              : s >= 40 ? 'rgba(245,158,11,0.1)'
              : 'rgba(239,68,68,0.1)',
            border: `1px solid ${s >= 70
              ? 'rgba(16,185,129,0.3)'
              : s >= 40 ? 'rgba(245,158,11,0.3)'
              : 'rgba(239,68,68,0.3)'}`,
          }}
        >
          {label}
        </div>
        <p className="trust-label" style={{ marginTop: 6 }}>
          {recommendation}
        </p>
      </div>
    </div>
  );
}
