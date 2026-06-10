import jsPDF from 'jspdf';

/* ─────────────────────────────────────────────────────────────────────────── *
 *  PhantomaShield — Standard 7-Section Clinical Radiology Report PDF
 * ─────────────────────────────────────────────────────────────────────────── */

const C = {
  bg:    [8,   11,  26 ], panel: [14, 18, 40], border: [30, 38, 70],
  cyan:  [0,  229, 255 ], green: [16,185,129], amber:  [245,158,11],
  red:   [239, 68,  68 ], blue:  [99, 102,241],
  pri:   [240,240, 255 ], sec:   [140,148,180], muted: [70, 80,120],
};

const f  = (doc, rgb) => doc.setFillColor(...rgb);
const s  = (doc, rgb) => doc.setDrawColor(...rgb);
const t  = (doc, rgb) => doc.setTextColor(...rgb);
const W  = 210, H = 297, M = 14, COL = W - 2 * M;

function bg(doc)       { f(doc, C.bg);  doc.rect(0,0,W,H,'F'); }
function hLine(doc, y) { s(doc,C.border); doc.setLineWidth(0.25); doc.line(M,y,W-M,y); }

function footer(doc, pageH) {
  f(doc,[8,12,30]); doc.rect(0,pageH-11,W,11,'F');
  t(doc,C.muted); doc.setFontSize(6); doc.setFont('helvetica','normal');
  doc.text('PhantomaShield  |  AI Clinical Radiology Report  |  PRELIMINARY — NOT FOR CLINICAL USE WITHOUT RADIOLOGIST REVIEW', M, pageH-5);
  doc.text(`Page ${doc.getCurrentPageInfo().pageNumber}`, W-M, pageH-5, {align:'right'});
}

function newPage(doc) {
  footer(doc, H);
  doc.addPage();
  bg(doc);
  return M + 8;
}

function overflow(doc, y, need=20) {
  return y + need > H - 22 ? newPage(doc) : y;
}

// Section header bar
function secBar(doc, y, num, title) {
  f(doc,[16,22,54]); doc.roundedRect(M, y, COL, 8, 1, 1, 'F');
  f(doc,C.cyan);     doc.roundedRect(M, y, 3, 8, 1, 1, 'F');
  // Number badge
  f(doc,[0,180,210]); doc.circle(M+10, y+4, 3.5, 'F');
  t(doc,[8,11,26]); doc.setFontSize(6.5); doc.setFont('helvetica','bold');
  doc.text(String(num), M+10, y+5.5, {align:'center'});
  t(doc,C.cyan); doc.setFontSize(7.5); doc.setFont('helvetica','bold');
  doc.text(title.toUpperCase(), M+17, y+5.5);
  return y + 12;
}

function bodyTxt(doc, x, y, text, maxW) {
  t(doc,C.sec); doc.setFontSize(8); doc.setFont('helvetica','normal');
  const lines = doc.splitTextToSize(String(text||'—'), maxW);
  doc.text(lines, x, y);
  return y + lines.length * 4.5;
}

function labelVal(doc, lx, y, label, value, maxW) {
  t(doc,C.muted); doc.setFontSize(6); doc.setFont('helvetica','bold');
  doc.text(label.toUpperCase(), lx, y);
  return bodyTxt(doc, lx, y+4, value, maxW) + 2;
}

// ── Info grid (2-column) ──────────────────────────────────────────────────────
function infoGrid(doc, y, items) {
  const cW = (COL-4) / 2;
  let col = 0, rowY = y;
  items.forEach(([lbl, val]) => {
    if (!val || val === 'N/A' || val === 'None') return;
    const x = M + col * (cW + 4);
    overflow(doc, rowY);
    f(doc,[16,20,46]); doc.roundedRect(x, rowY, cW, 14, 1, 1, 'F');
    t(doc,C.muted); doc.setFontSize(5.5); doc.setFont('helvetica','bold');
    doc.text(lbl.toUpperCase(), x+3, rowY+5);
    t(doc,C.sec); doc.setFontSize(8); doc.setFont('helvetica','normal');
    doc.text(doc.splitTextToSize(String(val), cW-6), x+3, rowY+10);
    col++;
    if (col === 2) { col = 0; rowY += 16; }
  });
  return col === 0 ? rowY : rowY + 16;
}

// ── Finding row ───────────────────────────────────────────────────────────────
function findingRow(doc, y, label, value) {
  if (!value || value === 'N/A') return y;
  y = overflow(doc, y, 12);
  hLine(doc, y);
  y += 3;
  t(doc,C.muted); doc.setFontSize(6); doc.setFont('helvetica','bold');
  doc.text(label.toUpperCase(), M+2, y+3);
  t(doc,C.sec); doc.setFontSize(8); doc.setFont('helvetica','normal');
  const lines = doc.splitTextToSize(value, COL - 58);
  doc.text(lines, M+58, y+3);
  return y + Math.max(6, lines.length * 4.5) + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
export async function generateRadiologyPdf(reportData, validationResult = null) {
  const doc = new jsPDF({ orientation:'portrait', unit:'mm', format:'a4' });
  const ts  = new Date().toLocaleString();
  const rid = `RAD-${Date.now().toString(36).toUpperCase()}`;

  const rep   = reportData?.report  || {};
  const pi    = rep.patient_info    || {};
  const si    = rep.study_info      || {};
  const fi    = rep.findings        || {};
  const impr  = rep.impression      || [];
  const label = reportData?.label   || 'UNKNOWN';
  const trust = reportData?.trust_score ?? 0;
  const conf  = reportData?.confidence  ?? 0;
  const genBy = reportData?.generated_by|| 'AI';
  const fname = reportData?.filename || 'unknown.dcm';

  const vC    = label === 'ORIGINAL' ? C.green : label === 'TAMPERED' ? C.amber : C.red;
  const tC    = trust >= 70 ? C.green : trust >= 40 ? C.amber : C.red;

  // ── Page 1 ──────────────────────────────────────────────────────────────────
  bg(doc);

  // ── Top header band ──────────────────────────────────────────────────────────
  f(doc,[10,14,36]); doc.rect(0,0,W,34,'F');
  f(doc,C.cyan);     doc.rect(0,0,4,34,'F');

  t(doc,C.cyan); doc.setFontSize(16); doc.setFont('helvetica','bold');
  doc.text('PhantomaShield', 10, 12);
  t(doc,C.pri);  doc.setFontSize(10); doc.setFont('helvetica','bold');
  doc.text('Clinical Radiology Report — Standard 7-Section Format', 10, 20);
  t(doc,C.muted); doc.setFontSize(6.5); doc.setFont('helvetica','normal');
  doc.text('AI-ASSISTED PRELIMINARY REPORT — MUST BE REVIEWED BY LICENSED RADIOLOGIST BEFORE CLINICAL USE', 10, 27);

  t(doc,C.muted); doc.setFontSize(6.5);
  doc.text(`Report ID: ${rid}`,  W-M, 10, {align:'right'});
  doc.text(`Generated: ${ts}`,   W-M, 16, {align:'right'});
  doc.text(`File: ${fname}`,     W-M, 22, {align:'right'});
  doc.text(`AI Engine: ${genBy}`,W-M, 28, {align:'right'});

  let y = 40;

  // ── Forensic verdict panel ────────────────────────────────────────────────
  f(doc,[14,18,44]); doc.roundedRect(M,y,COL,22,2,2,'F');
  s(doc,vC);        doc.setLineWidth(1.2); doc.roundedRect(M,y,COL,22,2,2,'D');
  t(doc,vC); doc.setFontSize(13); doc.setFont('helvetica','bold');
  doc.text(`FORENSIC STATUS: ${label}`, M+6, y+9);
  t(doc,C.sec); doc.setFontSize(7.5); doc.setFont('helvetica','normal');
  doc.text(rep.integrity_note||'', M+6, y+16, {maxWidth: COL-55});
  // Trust circle
  const cx=W-M-16, cy=y+11;
  s(doc,tC); doc.setLineWidth(2); doc.circle(cx,cy,10,'D');
  t(doc,tC); doc.setFontSize(9); doc.setFont('helvetica','bold');
  doc.text(`${Math.round(trust)}%`,cx,cy+3,{align:'center'});
  t(doc,C.muted); doc.setFontSize(5.5); doc.setFont('helvetica','normal');
  doc.text('TRUST',cx,cy+8,{align:'center'});
  y += 28;

  // ── 4-metric strip ────────────────────────────────────────────────────────
  const mW = (COL-9)/4;
  [
    {label:'Report Type',    value: rep.report_type||'—',             color:C.cyan},
    {label:'Confidence',     value: `${conf.toFixed(1)}%`,            color:conf>=70?C.green:conf>=40?C.amber:C.red},
    {label:'Trust Score',    value: `${Math.round(trust)}%`,          color:tC},
    {label:'Auth. Status',   value: label,                            color:vC},
  ].forEach(({label:lbl,value,color},i)=>{
    const mx = M + i*(mW+3);
    f(doc,[16,22,50]); doc.roundedRect(mx,y,mW,16,1,1,'F');
    t(doc,color); doc.setFontSize(9); doc.setFont('helvetica','bold');
    doc.text(value, mx+mW/2, y+8, {align:'center'});
    t(doc,C.muted); doc.setFontSize(5.5); doc.setFont('helvetica','normal');
    doc.text(lbl.toUpperCase(), mx+mW/2, y+13, {align:'center'});
  });
  y += 21;

  // ── DICOM images (if available) ────────────────────────────────────────────
  if (validationResult?.original_image) {
    const iW = validationResult.heatmap_image ? (COL-5)/2 : COL;
    const iH = 52;
    y = overflow(doc, y, iH+10);
    f(doc,[14,18,42]); doc.roundedRect(M,y,iW,iH,2,2,'F');
    t(doc,C.sec); doc.setFontSize(6); doc.setFont('helvetica','bold');
    doc.text('ORIGINAL DICOM IMAGE', M+iW/2, y+5, {align:'center'});
    try { doc.addImage(`data:image/png;base64,${validationResult.original_image}`,'PNG',M+2,y+7,iW-4,iH-10); } catch(_){}
    if (validationResult.heatmap_image) {
      const hx=M+iW+5;
      f(doc,[14,18,42]); doc.roundedRect(hx,y,iW,iH,2,2,'F');
      t(doc,C.amber);doc.setFontSize(6);doc.setFont('helvetica','bold');
      doc.text('GRAD-CAM HEATMAP',hx+iW/2,y+5,{align:'center'});
      try { doc.addImage(`data:image/png;base64,${validationResult.heatmap_image}`,'PNG',hx+2,y+7,iW-4,iH-10); } catch(_){}
    }
    y += iH + 8;
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 1 — Patient Information
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 50);
  y = secBar(doc, y, 1, '🧾  Patient Information');
  y = infoGrid(doc, y, [
    ['Patient Name',     pi.patient_name],
    ['Age',             pi.age],
    ['Sex',             pi.sex],
    ['Patient ID / MRN',pi.patient_id],
    ['Referring Doctor', pi.referring_doctor],
    ['Hospital',        pi.hospital],
  ]);
  y += 5;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 2 — Study Information
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 50);
  y = secBar(doc, y, 2, '📅  Study Information');
  y = infoGrid(doc, y, [
    ['Study Date',      si.study_date],
    ['Study Time',      si.study_time],
    ['Modality',        si.modality],
    ['Body Part',       si.body_part],
    ['Accession No.',   si.accession_number],
  ]);
  y += 5;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 3 — Clinical Indication
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 30);
  y = secBar(doc, y, 3, '⚙️  Clinical Indication');
  y = bodyTxt(doc, M+4, y, rep.clinical_indication, COL-8);
  y += 6;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 4 — Technique / Procedure
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 30);
  y = secBar(doc, y, 4, '🛠️  Technique / Procedure');
  y = bodyTxt(doc, M+4, y, rep.technique, COL-8);
  y += 6;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 5 — Findings (MOST IMPORTANT)
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 20);
  y = secBar(doc, y, 5, '🔍  Findings  (Detailed Observations)');
  y = findingRow(doc, y, 'Image Quality',        fi.image_quality);
  y = findingRow(doc, y, 'Lungs / Parenchyma',   fi.lungs);
  y = findingRow(doc, y, 'Pleura',               fi.pleura);
  y = findingRow(doc, y, 'Mediastinum',          fi.mediastinum);
  y = findingRow(doc, y, 'Heart & Great Vessels', fi.heart_vessels);
  y = findingRow(doc, y, 'Bones & Soft Tissue',  fi.bones_soft_tissue);
  if (fi.other && fi.other !== 'N/A') y = findingRow(doc, y, 'Other', fi.other);
  y += 6;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 6 — Impression (Conclusion)
  // ═══════════════════════════════════════════════════════════════════════
  y = overflow(doc, y, 25);
  y = secBar(doc, y, 6, '🧠  Impression  (Conclusion)');
  impr.forEach((imp, i) => {
    y = overflow(doc, y, 12);
    t(doc,C.cyan); doc.setFontSize(7.5); doc.setFont('helvetica','bold');
    doc.text(`${i+1}.`, M+3, y+1);
    t(doc,C.sec); doc.setFontSize(8); doc.setFont('helvetica','normal');
    const lines = doc.splitTextToSize(imp, COL-12);
    doc.text(lines, M+9, y+1);
    y += lines.length*4.5 + 3;
  });
  y += 4;

  // ═══════════════════════════════════════════════════════════════════════
  //  SECTION 7 — Recommendations
  // ═══════════════════════════════════════════════════════════════════════
  if (rep.recommendation) {
    y = overflow(doc, y, 25);
    y = secBar(doc, y, 7, '⚠️  Recommendations');
    t(doc,C.cyan); doc.setFontSize(8.5); doc.setFont('helvetica','italic');
    const rLines = doc.splitTextToSize(rep.recommendation, COL-8);
    doc.text(rLines, M+4, y);
    y += rLines.length*5 + 6;
  }

  // ── Disclaimer ────────────────────────────────────────────────────────────
  y = overflow(doc, y, 20);
  f(doc,[28,22,8]); doc.roundedRect(M,y,COL,16,2,2,'F');
  s(doc,C.amber);   doc.setLineWidth(0.6); doc.roundedRect(M,y,COL,16,2,2,'D');
  t(doc,C.amber); doc.setFontSize(6.5); doc.setFont('helvetica','bold');
  doc.text('⚠  DISCLAIMER', M+5, y+5);
  t(doc,C.sec); doc.setFontSize(7); doc.setFont('helvetica','normal');
  const dlines = doc.splitTextToSize(rep.disclaimer||'AI-generated preliminary report. Must be reviewed by a licensed radiologist.', COL-10);
  doc.text(dlines, M+5, y+11);
  y += 19;

  // ── Forensic Signals Appendix ─────────────────────────────────────────────
  if (validationResult?.forensics) {
    const fs = validationResult.forensics;
    y = overflow(doc, y+4, 50);
    f(doc,[16,20,50]); doc.roundedRect(M,y,COL,8,1,1,'F');
    f(doc,[245,158,11]); doc.roundedRect(M,y,3,8,1,1,'F');
    t(doc,C.amber); doc.setFontSize(7); doc.setFont('helvetica','bold');
    doc.text('APPENDIX — FORENSIC SIGNAL ANALYSIS', M+8, y+5.5);
    y += 11;
    const sigW=(COL-6)/2;
    [
      ['FFT GAN Frequency',  `${((fs.fft_ai_score||0)*100).toFixed(0)}%`,   `Spectral slope: ${fs.spectral_slope?.toFixed(3)??'N/A'}`],
      ['Noise Residual',      `${((fs.noise_ai_score||0)*100).toFixed(0)}%`, `Noise std: ${fs.noise_std?.toFixed(5)??'N/A'}`],
      ['Texture/DCT',         `${((fs.texture_ai_score||0)*100).toFixed(0)}%`,`DCT uniform: ${fs.dct_uniform?'Yes ⚠':'No ✓'}`],
      ['Ensemble AI Risk',    `${((fs.ai_composite||0)*100).toFixed(0)}%`,   `Tamper: ${((fs.tamper_composite||0)*100).toFixed(0)}%`],
    ].forEach((sig, i) => {
      const r2=Math.floor(i/2), c2=i%2;
      const sx=M+c2*(sigW+6), sy=y+r2*18;
      f(doc,[16,20,46]); doc.roundedRect(sx,sy,sigW,15,2,2,'F');
      const sc=parseFloat(sig[1])>45?C.red:parseFloat(sig[1])>25?C.amber:C.green;
      t(doc,C.muted); doc.setFontSize(5.5); doc.setFont('helvetica','bold');
      doc.text(sig[0].toUpperCase(), sx+4, sy+5);
      t(doc,sc); doc.setFontSize(9.5); doc.setFont('helvetica','bold');
      doc.text(sig[1], sx+sigW-4, sy+10,{align:'right'});
      t(doc,C.muted); doc.setFontSize(6); doc.setFont('helvetica','normal');
      doc.text(sig[2], sx+4, sy+12);
    });
  }

  // ── Footers on every page ─────────────────────────────────────────────────
  const totalPages = doc.getNumberOfPages();
  for (let p=1; p<=totalPages; p++) {
    doc.setPage(p);
    footer(doc, H);
  }

  doc.save(`phantomashield_radiology_report_${Date.now()}.pdf`);
}
