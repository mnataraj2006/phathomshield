"""
data_store.py — PhantomaShield Local Data Storage
===================================================
Provides simple, offline-first persistence for validation and recovery cases.

Folder layout (relative to this file's parent directory):
    data/
    ├── uploads/
    ├── validation/
    │   ├── images/
    │   ├── reports/
    │   └── validation.json
    └── recovery/
        ├── images/
        ├── reports/
        └── recovery.json

All public functions are thread-safe via a module-level lock.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("phantomashield.data_store")

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # backend/
DATA_ROOT   = _HERE.parent / "data"              # project root / data/

UPLOADS_DIR        = DATA_ROOT / "uploads"

VALIDATION_DIR     = DATA_ROOT / "validation"
VALIDATION_IMAGES  = VALIDATION_DIR / "images"
VALIDATION_REPORTS = VALIDATION_DIR / "reports"
VALIDATION_JSON    = VALIDATION_DIR / "validation.json"

RECOVERY_DIR       = DATA_ROOT / "recovery"
RECOVERY_IMAGES    = RECOVERY_DIR / "images"
RECOVERY_REPORTS   = RECOVERY_DIR / "reports"
RECOVERY_JSON      = RECOVERY_DIR / "recovery.json"

_lock = threading.Lock()


# ─── Bootstrap ────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    """Create the full data folder structure if it does not already exist."""
    for d in (
        UPLOADS_DIR,
        VALIDATION_IMAGES, VALIDATION_REPORTS,
        RECOVERY_IMAGES,   RECOVERY_REPORTS,
    ):
        d.mkdir(parents=True, exist_ok=True)
    # Touch JSON index files so they always exist
    for jf in (VALIDATION_JSON, RECOVERY_JSON):
        if not jf.exists():
            jf.write_text("[]", encoding="utf-8")


_ensure_dirs()


# ─── JSON helpers ─────────────────────────────────────────────────────────────
def _read_json(path: Path) -> list[dict]:
    """Read a JSON array file; return empty list on any error."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; returning []", path)
        return []


def _write_json(path: Path, data: list[dict]) -> None:
    """Atomically write a JSON array to disk."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ─── Image helpers ────────────────────────────────────────────────────────────
def _save_b64_image(b64_string: str | None, dest_dir: Path, stem: str) -> str | None:
    """
    Decode a base64-encoded PNG and write it to *dest_dir/<stem>.png*.
    Returns the relative path (from DATA_ROOT) or None if *b64_string* is falsy.
    """
    if not b64_string:
        return None
    try:
        img_bytes = base64.b64decode(b64_string)
        out_path = dest_dir / f"{stem}.png"
        out_path.write_bytes(img_bytes)
        return str(out_path.relative_to(DATA_ROOT))
    except Exception as exc:
        logger.warning("Could not save image for %s: %s", stem, exc)
        return None


# ─── Validation Storage ───────────────────────────────────────────────────────
def save_validation(
    *,
    case_id: str,
    filename: str,
    label: str,
    trust_score: int,
    confidence: float,
    image_b64: str | None = None,
    heatmap_b64: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """
    Persist a validation result entry and return the stored record.

    Parameters
    ----------
    case_id   : Shared case ID (use the same value for the recovery record).
    filename  : Original DICOM filename.
    label     : ORIGINAL | TAMPERED | AI-GENERATED
    trust_score: 0-100 integer.
    confidence : 0-100 float.
    image_b64  : Base64 PNG of the original image (optional).
    heatmap_b64: Base64 PNG of the Grad-CAM heatmap (optional).
    extra      : Any additional metadata to store alongside the record.
    """
    import datetime as _dt

    image_path   = _save_b64_image(image_b64,   VALIDATION_IMAGES, f"{case_id}_orig")
    heatmap_path = _save_b64_image(heatmap_b64, VALIDATION_IMAGES, f"{case_id}_heatmap")

    # Friendly status mapping
    status_map = {
        "ORIGINAL":     "Safe",
        "TAMPERED":     "Tampered",
        "AI-GENERATED": "AI-Generated",
    }
    status = status_map.get(label.upper(), label)

    record: dict[str, Any] = {
        "id":           case_id,
        "file_name":    filename,
        "status":       status,
        "label":        label.upper(),
        "trust_score":  trust_score,
        "confidence":   round(confidence, 2),
        "date":         _dt.date.today().isoformat(),
        "timestamp":    _dt.datetime.utcnow().isoformat() + "Z",
        "image_path":   image_path,
        "heatmap_path": heatmap_path,
        "report_path":  None,            # populated by save_report() if needed
    }
    if extra:
        record.update(extra)

    with _lock:
        entries = _read_json(VALIDATION_JSON)
        # Avoid duplicates: update in-place if ID already exists
        ids = {e.get("id") for e in entries}
        if case_id not in ids:
            entries.append(record)
        else:
            entries = [record if e.get("id") == case_id else e for e in entries]
        _write_json(VALIDATION_JSON, entries)

    logger.info("Saved validation record: %s → %s (trust=%d)", case_id, status, trust_score)
    return record


# ─── Recovery Storage ─────────────────────────────────────────────────────────
def save_recovery(
    *,
    case_id: str,
    filename: str,
    recovered: bool,
    corruption_type: str = "unknown",
    severity: str = "unknown",
    affected_pct: float = 0.0,
    quality_score: float | None = None,
    image_b64: str | None = None,
    recovered_image_b64: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """
    Persist a recovery result entry and return the stored record.

    Parameters
    ----------
    case_id             : Shared case ID (matches the validation record).
    filename            : Original DICOM filename.
    recovered           : Whether recovery succeeded.
    corruption_type     : Type of corruption detected.
    severity            : low | medium | high | critical
    affected_pct        : Percentage of pixels affected.
    quality_score       : 0-1 float PSNR-derived quality estimate (optional).
    image_b64           : Base64 PNG of the corrupted image (optional).
    recovered_image_b64 : Base64 PNG of the recovered image (optional).
    extra               : Any additional metadata to store alongside the record.
    """
    import datetime as _dt

    orig_path  = _save_b64_image(image_b64,           RECOVERY_IMAGES, f"{case_id}_corrupted")
    recon_path = _save_b64_image(recovered_image_b64, RECOVERY_IMAGES, f"{case_id}_recovered")

    record: dict[str, Any] = {
        "id":               case_id,
        "file_name":        filename,
        "recovered":        recovered,
        "corruption_type":  corruption_type,
        "severity":         severity,
        "affected_pct":     round(affected_pct, 2),
        "quality_score":    round(quality_score, 4) if quality_score is not None else None,
        "date":             _dt.date.today().isoformat(),
        "timestamp":        _dt.datetime.utcnow().isoformat() + "Z",
        "image_path":       orig_path,
        "recovered_image_path": recon_path,
        "report_path":      None,
    }
    if extra:
        record.update(extra)

    with _lock:
        entries = _read_json(RECOVERY_JSON)
        ids = {e.get("id") for e in entries}
        if case_id not in ids:
            entries.append(record)
        else:
            entries = [record if e.get("id") == case_id else e for e in entries]
        _write_json(RECOVERY_JSON, entries)

    logger.info("Saved recovery record: %s | recovered=%s | quality=%s", case_id, recovered, quality_score)
    return record


# ─── Load Functions ───────────────────────────────────────────────────────────
def load_validation() -> list[dict]:
    """Return all validation records, newest first."""
    with _lock:
        entries = _read_json(VALIDATION_JSON)
    return sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)


def load_recovery() -> list[dict]:
    """Return all recovery records, newest first."""
    with _lock:
        entries = _read_json(RECOVERY_JSON)
    return sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)


# ─── Merge / Cases ────────────────────────────────────────────────────────────
def load_cases() -> list[dict]:
    """
    Merge validation and recovery records by shared `id`.

    Returns a list of combined case dicts, newest first.
    Each dict contains all validation fields plus any matching recovery fields.
    Cases that only have a recovery record are also included.
    """
    val_map  = {e["id"]: e for e in load_validation()}
    rec_map  = {e["id"]: e for e in load_recovery()}

    all_ids_ordered: list[str] = []
    seen: set[str] = set()

    # Collect IDs in reverse-chronological order (validation first, then recovery-only)
    for entry in sorted(
        list(val_map.values()) + [r for r in rec_map.values() if r["id"] not in val_map],
        key=lambda e: e.get("timestamp", ""),
        reverse=True,
    ):
        cid = entry["id"]
        if cid not in seen:
            all_ids_ordered.append(cid)
            seen.add(cid)

    cases: list[dict] = []
    for cid in all_ids_ordered:
        v = val_map.get(cid, {})
        r = rec_map.get(cid, {})

        # Prefer image from validation if available, fall back to recovery
        preview_image = v.get("image_path") or r.get("image_path") or r.get("recovered_image_path")

        case: dict[str, Any] = {
            "id":               cid,
            "file_name":        v.get("file_name") or r.get("file_name", "unknown"),
            "date":             v.get("date") or r.get("date", ""),
            "timestamp":        v.get("timestamp") or r.get("timestamp", ""),
            # ── Validation fields
            "status":           v.get("status"),
            "label":            v.get("label"),
            "trust_score":      v.get("trust_score"),
            "confidence":       v.get("confidence"),
            "heatmap_path":     v.get("heatmap_path"),
            # ── Recovery fields
            "recovered":        r.get("recovered"),
            "corruption_type":  r.get("corruption_type"),
            "severity":         r.get("severity"),
            "affected_pct":     r.get("affected_pct"),
            "quality_score":    r.get("quality_score"),
            "recovered_image_path": r.get("recovered_image_path"),
            # ── Shared
            "image_path":       preview_image,
            "report_path":      v.get("report_path") or r.get("report_path"),
            # ── Module flags (useful for UI filtering)
            "has_validation":   bool(v),
            "has_recovery":     bool(r),
        }
        cases.append(case)

    return cases
