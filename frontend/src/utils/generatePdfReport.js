import jsPDF from 'jspdf';

/* ─────────────────────────────────────────────────────────────────────────── *
 *  PhantomaShield – PDF Report Generator
 *  Uses jsPDF only (no html2canvas) for fast, crisp, vector-based output.
 * ─────────────────────────────────────────────────────────────────────────── */

// ── Palette (dark-mode brand colours) ──────────────────────────────────────
const C = {
  bg:        [10,  12,  20 ],
  panel:     [16,  20,  38 ],
  border:    [32,  40,  72 ],
  cyan:      [0,   229, 255 ],
  green:     [16,  185, 129 ],
  amber:     [245, 158, 11  ],
  red:       [239, 68,  68  ],
  purple:    [139, 92,  246 ],
  textPri:   [240, 240, 255 ],
  textSec:   [140, 148, 180 ],
  textMuted: [80,  90,  130 ],
  white:     [255, 255, 255 ],
};

// ── Friendly DICOM tag names ────────────────────────────────────────────────
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

// ── Helper – set fill colour ───────────────────────────────────────────────
function fill(doc, rgb) { doc.setFillColor(...rgb); }
function stroke(doc, rgb) { doc.setDrawColor(...rgb); }
function text(doc, rgb) { doc.setTextColor(...rgb); }

// ── Helper – rounded rect (filled) ────────────────────────────────────────
function rRect(doc, x, y, w, h, r, fillRgb, strokeRgb) {
  if (fillRgb) fill(doc, fillRgb);
  if (strokeRgb) stroke(doc, strokeRgb); else doc.setDrawColor(0);
  const style = fillRgb && strokeRgb ? 'FD' : fillRgb ? 'F' : 'D';
  doc.roundedRect(x, y, w, h, r, r, style);
}

// ── Helper – confidence / signal bar ──────────────────────────────────────
function drawBar(doc, x, y, w, pct, barRgb) {
  const h = 4;
  rRect(doc, x, y, w, h, 1, [28, 34, 60]);
  if (pct > 0) rRect(doc, x, y, w * Math.min(pct / 100, 1), h, 1, barRgb);
}

// ── Helper – section header ────────────────────────────────────────────────
function sectionHeader(doc, y, label, pageW, margin) {
  rRect(doc, margin, y, pageW - 2 * margin, 9, 2, [20, 26, 52]);
  text(doc, C.cyan);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'bold');
  doc.text(label, margin + 5, y + 6);
  return y + 13;
}

// ── Helper – page background ───────────────────────────────────────────────
function pageBg(doc, pageW, pageH) {
  fill(doc, C.bg);
  doc.rect(0, 0, pageW, pageH, 'F');
}

// ── Helper – draw verdict badge ───────────────────────────────────────────
function verdictColor(label) {
  const l = (label || '').toUpperCase();
  if (l.includes('ORIGINAL') || l.includes('REAL')) return C.green;
  if (l.includes('TAMPERED')) return C.amber;
  return C.red;
}

// ── Helper – load base64 image as Data URL ─────────────────────────────────
function b64url(b64) { return `data:image/png;base64,${b64}`; }

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
export async function generatePdfReport(result, fileName = null) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageW = 210;
  const pageH = 297;
  const margin = 14;
  const col = pageW - 2 * margin;
  const ts = new Date().toLocaleString();
  const reportId = `PS-${Date.now().toString(36).toUpperCase()}`;

  // Destructure result fields with safe defaults
  const label       = result?.label       || 'UNKNOWN';
  const confidence  = result?.confidence  ?? 0;
  const trust       = result?.trust_score ?? 0;
  const forensics   = result?.forensics   || {};
  const metadata    = result?.metadata    || {};
  const tags        = metadata?.tags      || {};
  const integrity   = metadata?.integrity || {};

  // ── Page 1 ──────────────────────────────────────────────────────────────

  pageBg(doc, pageW, pageH);

  // ── Header bar ────────────────────────────────────────────────────────────
  fill(doc, [12, 18, 42]);
  doc.rect(0, 0, pageW, 28, 'F');
  // Accent stripe
  fill(doc, C.cyan);
  doc.rect(0, 0, 4, 28, 'F');

  text(doc, C.cyan);
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text('PhantomaShield', 10, 12);

  text(doc, C.textSec);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text('AI Medical Image Forensics Platform', 10, 19);

  text(doc, C.textMuted);
  doc.setFontSize(7);
  doc.text(`Report ID: ${reportId}`, pageW - 14, 10, { align: 'right' });
  doc.text(`Generated: ${ts}`, pageW - 14, 16, { align: 'right' });
  doc.text('Classification: CONFIDENTIAL', pageW - 14, 22, { align: 'right' });

  let y = 35;

  // ── Verdict Panel ─────────────────────────────────────────────────────────
  const vColor = verdictColor(label);
  rRect(doc, margin, y, col, 24, 3, [16, 20, 42], vColor);

  text(doc, vColor);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(label, margin + 6, y + 9);

  text(doc, C.textSec);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text('Forensic Detection Verdict', margin + 6, y + 16);

  // Trust score circle (right side)
  const tColor = trust >= 70 ? C.green : trust >= 40 ? C.amber : C.red;
  const cx2 = pageW - margin - 20, cy2 = y + 12;
  stroke(doc, tColor);
  doc.setLineWidth(2);
  doc.circle(cx2, cy2, 10, 'D');
  text(doc, tColor);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.text(`${Math.round(trust)}%`, cx2, cy2 + 3, { align: 'center' });
  text(doc, C.textMuted);
  doc.setFontSize(6);
  doc.setFont('helvetica', 'normal');
  doc.text('TRUST', cx2, cy2 + 8, { align: 'center' });

  y += 30;

  // ── Key Metrics Row ────────────────────────────────────────────────────────
  const metrics = [
    { label: 'Confidence',        val: `${confidence.toFixed(1)}%`,   sub: 'Model confidence', color: confidence >= 70 ? C.green : confidence >= 40 ? C.amber : C.red },
    { label: 'Metadata Integrity', val: integrity?.status || '—',      sub: `${metadata?.present_count ?? '—'}/${metadata?.total_count ?? '—'} tags`, color: integrity?.status === 'VALID' ? C.green : integrity?.status === 'SUSPICIOUS' ? C.amber : C.red },
    { label: 'AI Risk Score',      val: `${Math.round((forensics.ai_composite ?? 0) * 100)}%`, sub: 'Ensemble composite', color: (forensics.ai_composite ?? 0) > 0.45 ? C.red : (forensics.ai_composite ?? 0) > 0.25 ? C.amber : C.green },
    { label: 'Tamper Risk',        val: `${Math.round((forensics.tamper_composite ?? 0) * 100)}%`, sub: 'Tamper composite',  color: (forensics.tamper_composite ?? 0) > 0.45 ? C.red : (forensics.tamper_composite ?? 0) > 0.25 ? C.amber : C.green },
  ];

  const mW = (col - 9) / 4;
  metrics.forEach((m, i) => {
    const mx = margin + i * (mW + 3);
    rRect(doc, mx, y, mW, 22, 2, [18, 24, 48]);
    text(doc, m.color);
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text(m.val, mx + mW / 2, y + 11, { align: 'center' });
    text(doc, C.textSec);
    doc.setFontSize(6);
    doc.setFont('helvetica', 'bold');
    doc.text(m.label.toUpperCase(), mx + mW / 2, y + 16, { align: 'center' });
    text(doc, C.textMuted);
    doc.setFontSize(5.5);
    doc.setFont('helvetica', 'normal');
    doc.text(m.sub, mx + mW / 2, y + 20, { align: 'center' });
  });

  y += 28;

  // ── Images Section (Original + Heatmap) ───────────────────────────────────
  y = sectionHeader(doc, y, '  DICOM IMAGE ANALYSIS  |  Grad-CAM Tamper Heatmap', pageW, margin);

  const imgW = (col - 6) / 2;
  const imgH = 60;

  // Original image card
  rRect(doc, margin, y, imgW, imgH, 2, [16, 20, 42]);
  text(doc, C.textSec);
  doc.setFontSize(7);
  doc.setFont('helvetica', 'bold');
  doc.text('ORIGINAL DICOM IMAGE', margin + imgW / 2, y + 6, { align: 'center' });

  if (result?.original_image) {
    try {
      doc.addImage(b64url(result.original_image), 'PNG', margin + 2, y + 8, imgW - 4, imgH - 12);
    } catch (_) { /* skip if image fails */ }
  }

  // Heatmap image card
  const hx = margin + imgW + 6;
  rRect(doc, hx, y, imgW, imgH, 2, [16, 20, 42]);
  text(doc, C.amber);
  doc.setFontSize(7);
  doc.setFont('helvetica', 'bold');
  doc.text('GRAD-CAM TAMPER HEATMAP', hx + imgW / 2, y + 6, { align: 'center' });

  const heatmapSrc = result?.heatmap_image || result?.original_image;
  if (heatmapSrc) {
    try {
      doc.addImage(b64url(heatmapSrc), 'PNG', hx + 2, y + 8, imgW - 4, imgH - 12);
    } catch (_) { /* skip if image fails */ }
  }

  // Legend strip
  text(doc, C.textMuted);
  doc.setFontSize(5.5);
  doc.setFont('helvetica', 'normal');
  doc.text('Red = High Suspicion  |  Orange = Moderate  |  Green = Normal', hx + imgW / 2, y + imgH - 1, { align: 'center' });

  y += imgH + 6;

  // ── Forensic Signal Analysis ───────────────────────────────────────────────
  y = sectionHeader(doc, y, '  FORENSIC SIGNAL ANALYSIS', pageW, margin);

  const signals = [
    {
      icon: 'FFT Frequency Analysis',
      score: (forensics.fft_ai_score ?? 0) * 100,
      detail: `Spectral Slope: ${forensics.spectral_slope?.toFixed(3) ?? 'N/A'}`,
      note: forensics.spectral_slope > -0.5
        ? 'Too flat — AI signature'
        : forensics.spectral_slope < -3
          ? 'Too steep — synthetic smoothing'
          : 'Natural 1/f profile',
    },
    {
      icon: 'Noise Residual Analysis',
      score: (forensics.noise_ai_score ?? 0) * 100,
      detail: `Noise Std: ${forensics.noise_std?.toFixed(5) ?? 'N/A'}`,
      note: forensics.noise_std < 0.003 ? 'Too smooth — synthesized' : 'Natural scanner noise',
    },
    {
      icon: 'Texture / DCT Uniformity',
      score: (forensics.texture_ai_score ?? 0) * 100,
      detail: `DCT Uniform: ${forensics.dct_uniform ? 'Yes' : 'No'}`,
      note: forensics.dct_uniform ? 'Over-uniform — AI signature' : 'Natural DCT diversity',
    },
    {
      icon: 'Ensemble AI Risk',
      score: (forensics.ai_composite ?? 0) * 100,
      detail: `Tamper Risk: ${Math.round((forensics.tamper_composite ?? 0) * 100)}%`,
      note: (forensics.ai_composite ?? 0) > 0.45 ? 'HIGH AI RISK' : (forensics.ai_composite ?? 0) > 0.25 ? 'MODERATE RISK' : 'LOW RISK',
    },
  ];

  const sW = (col - 9) / 2;
  signals.forEach((sig, i) => {
    const row = Math.floor(i / 2);
    const col2 = i % 2;
    const sx = margin + col2 * (sW + 3);
    const sy = y + row * 22;
    rRect(doc, sx, sy, sW, 19, 2, [18, 24, 50]);

    // Bar color
    const barC = sig.score > 45 ? C.red : sig.score > 25 ? C.amber : C.green;

    text(doc, C.textSec);
    doc.setFontSize(6.5);
    doc.setFont('helvetica', 'bold');
    doc.text(sig.icon.toUpperCase(), sx + 4, sy + 5);

    drawBar(doc, sx + 4, sy + 7, sW - 35, sig.score, barC);

    text(doc, barC);
    doc.setFontSize(7);
    doc.setFont('helvetica', 'bold');
    doc.text(`${sig.score.toFixed(0)}%`, sx + sW - 30, sy + 11);

    text(doc, C.textMuted);
    doc.setFontSize(5.5);
    doc.setFont('helvetica', 'normal');
    doc.text(sig.detail, sx + 4, sy + 13);

    const noteC = sig.note.includes('AI') || sig.note.includes('smooth') || sig.note.includes('HIGH') ? C.red
      : sig.note.includes('Moderate') || sig.note.includes('MODERATE') ? C.amber : C.green;
    text(doc, noteC);
    doc.setFontSize(5.5);
    doc.text(`→ ${sig.note}`, sx + 4, sy + 17);
  });

  y += 2 * 22 + 4;

  // ── Metadata Integrity Warnings ────────────────────────────────────────────
  const aiInd  = integrity?.ai_indicators || [];
  const warns  = integrity?.warnings      || [];

  if (aiInd.length > 0 || warns.length > 0) {
    y = sectionHeader(doc, y, '  METADATA INTEGRITY FLAGS', pageW, margin);

    if (aiInd.length > 0) {
      rRect(doc, margin, y, col, 6 + aiInd.length * 5, 2, [30, 10, 10]);
      text(doc, C.red);
      doc.setFontSize(6.5);
      doc.setFont('helvetica', 'bold');
      doc.text('AI / Synthetic Generation Indicators', margin + 4, y + 5);
      aiInd.forEach((ind, i) => {
        text(doc, C.textSec);
        doc.setFontSize(6);
        doc.setFont('helvetica', 'normal');
        doc.text(`• ${ind}`, margin + 6, y + 10 + i * 5);
      });
      y += 6 + aiInd.length * 5 + 3;
    }

    if (warns.length > 0) {
      rRect(doc, margin, y, col, 6 + warns.length * 5, 2, [30, 24, 5]);
      text(doc, C.amber);
      doc.setFontSize(6.5);
      doc.setFont('helvetica', 'bold');
      doc.text('Metadata Warnings', margin + 4, y + 5);
      warns.forEach((w, i) => {
        text(doc, C.textSec);
        doc.setFontSize(6);
        doc.setFont('helvetica', 'normal');
        doc.text(`• ${w}`, margin + 6, y + 10 + i * 5);
      });
      y += 6 + warns.length * 5 + 3;
    }
  }

  // ── File Hash ─────────────────────────────────────────────────────────────
  if (integrity?.hash) {
    rRect(doc, margin, y, col, 8, 1, [14, 18, 40]);
    text(doc, C.textMuted);
    doc.setFontSize(6);
    doc.setFont('helvetica', 'normal');
    doc.text('File Hash (MD5): ', margin + 4, y + 5);
    text(doc, C.cyan);
    doc.setFont('courier', 'normal');
    doc.text(integrity.hash, margin + 32, y + 5);
    y += 11;
  }

  // ── DICOM Metadata Tags Table ──────────────────────────────────────────────
  // Move to new page if not enough space
  const tagEntries = Object.entries(tags);
  const neededHeight = 16 + tagEntries.length * 6;

  if (y + neededHeight > pageH - 20) {
    doc.addPage();
    pageBg(doc, pageW, pageH);
    y = 14;
  }

  y = sectionHeader(doc, y, '  DICOM METADATA TAGS', pageW, margin);

  // Table header
  rRect(doc, margin, y, col, 7, 1, [22, 30, 60]);
  const colWidths = [30, 45, 70, 22];
  const headers = ['Tag', 'Description', 'Value', 'Status'];
  let cx = margin + 2;
  headers.forEach((h, i) => {
    text(doc, C.cyan);
    doc.setFontSize(6);
    doc.setFont('helvetica', 'bold');
    doc.text(h, cx, y + 5);
    cx += colWidths[i];
  });
  y += 7;

  // Suspicion score badge
  if ((integrity?.suspicion_score ?? 0) > 0) {
    const ss = integrity.suspicion_score;
    const ssC = ss >= 35 ? C.red : C.amber;
    text(doc, ssC);
    doc.setFontSize(6);
    doc.setFont('helvetica', 'bold');
    const ssText = `Suspicion Score: ${ss}/100`;
    const tw = doc.getTextWidth(ssText);
    rRect(doc, pageW - margin - tw - 6, y - 7.5, tw + 4, 5, 1, undefined, ssC);
    doc.text(ssText, pageW - margin - 4, y - 5, { align: 'right' });
  }

  // Tag rows
  tagEntries.forEach(([tag, value], i) => {
    const rowBg = i % 2 === 0 ? [14, 18, 36] : [16, 20, 42];
    rRect(doc, margin, y, col, 6, 0, rowBg);

    const isPresent = value !== null && value !== undefined && value !== '';
    const friendly  = FRIENDLY_TAGS[tag] || 'Unknown Tag';
    const valStr    = isPresent ? String(value).slice(0, 38) : '—';
    const statusC   = isPresent ? C.green : C.red;

    cx = margin + 2;
    text(doc, C.textSec);
    doc.setFontSize(5.5);
    doc.setFont('courier', 'normal');
    doc.text(tag, cx, y + 4); cx += colWidths[0];

    doc.setFont('helvetica', 'normal');
    text(doc, C.textMuted);
    doc.text(friendly, cx, y + 4); cx += colWidths[1];

    text(doc, C.textPri);
    doc.text(valStr, cx, y + 4); cx += colWidths[2];

    text(doc, statusC);
    doc.setFont('helvetica', 'bold');
    doc.text(isPresent ? '✓ OK' : '✗ Miss', cx, y + 4);

    y += 6;

    // Overflow: add new page
    if (y > pageH - 18) {
      doc.addPage();
      pageBg(doc, pageW, pageH);
      y = 14;
    }
  });

  // ── Footer on every page ──────────────────────────────────────────────────
  const totalPages = doc.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    fill(doc, [10, 14, 32]);
    doc.rect(0, pageH - 10, pageW, 10, 'F');
    text(doc, C.textMuted);
    doc.setFontSize(6);
    doc.setFont('helvetica', 'normal');
    doc.text('PhantomaShield  |  AI Medical Image Forensics Platform  |  CONFIDENTIAL', margin, pageH - 4);
    doc.text(`Page ${p} / ${totalPages}`, pageW - margin, pageH - 4, { align: 'right' });
  }

  // ── Save ─────────────────────────────────────────────────────────────────
  const name = fileName || `phantomashield_report_${Date.now()}.pdf`;
  doc.save(name);
}
