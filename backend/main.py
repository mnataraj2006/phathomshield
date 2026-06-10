"""
main.py — PhantomaShield FastAPI Backend
=========================================
Endpoints:
  POST /validate   → Module 1: DICOM detection + metadata validation
  POST /recover    → Module 2: Corrupted file recovery
  GET  /result/{id}→ Fetch cached result (uses in-memory store for MVP)
  GET  /health     → Health check
"""
import io
import os
import uuid
import base64
import logging
import numpy as np
import pydicom
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dicom_loader import load_dicom
from preprocessor import to_rgb, array_to_base64_png
from detector import classify
from localizer import generate_heatmap_full
from validator import validate_metadata
from trust_score import compute_trust_score_full
from corruption_detector import detect_corruption
from recovery_engine import recover_image
from metadata_restorer import restore_metadata
from radiology_reporter import generate_radiology_report, RateLimitError
import data_store

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phantomashield")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhantomaShield API",
    description="AI-Powered DICOM Medical Image Integrity Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Configuration ───────────────────────────────────────────────────────
# allow_origins=["*"] is required for cross-origin requests from Vercel → Render.
# NOTE: allow_credentials must be False when using wildcard origins (browser spec).
# To restrict in future: set ALLOWED_ORIGINS env var to comma-separated URLs
# and change allow_credentials back to True.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_explicit_origins: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins.strip()
    else []
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_explicit_origins if _explicit_origins else ["*"],
    allow_credentials=bool(_explicit_origins),  # False when using wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result cache (use Redis/S3 in production)
_result_cache: dict = {}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ─── Activity Log ─────────────────────────────────────────────────────────────
import datetime as _dt
_activity_log: list = []   # max 50 entries
MAX_ACTIVITY = 50

def _log_activity(filename: str, action: str, result_id: str):
    """Append a live activity entry (action: ORIGINAL|TAMPERED|AI-GENERATED|RECOVERED|REPORT)."""
    _activity_log.append({
        "filename": filename,
        "action": action,
        "result_id": result_id,
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
    })
    # Keep only the most recent MAX_ACTIVITY entries
    if len(_activity_log) > MAX_ACTIVITY:
        del _activity_log[:-MAX_ACTIVITY]


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "PhantomaShield API"}


# ─── Module 1: Validate & Detect ──────────────────────────────────────────────
@app.post("/validate", tags=["Module 1 — Detection"])
async def validate_dicom(file: UploadFile = File(...)):
    """
    Analyze a DICOM file for authenticity, tamper localization, and metadata integrity.

    Returns:
    - label: ORIGINAL / TAMPERED / AI-GENERATED
    - confidence: 0-100
    - trust_score: 0-100
    - original_image: base64 PNG
    - heatmap_image: base64 PNG
    - metadata: tags, integrity, hash
    """
    # ── File validation
    if not (file.filename.lower().endswith(".dcm") or file.content_type in ("application/dicom", "application/octet-stream")):
        raise HTTPException(400, "Only .dcm DICOM files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")
    if len(file_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    result_id = str(uuid.uuid4())

    try:
        # ── Layer 1: Load DICOM
        logger.info("[%s] Loading DICOM: %s", result_id, file.filename)
        dicom_data = load_dicom(file_bytes)
        pixel_array = dicom_data["pixel_array"]
        decode_method = dicom_data.get("decode_method", "unknown")
        transfer_syntax = dicom_data.get("transfer_syntax", "unknown")

        if pixel_array is None:
            ts_msg = f" (Transfer Syntax: {transfer_syntax})" if transfer_syntax else ""
            raise HTTPException(
                422,
                f"Could not decode pixel data from this DICOM file{ts_msg}. "
                f"The file may use JPEG or JPEG-2000 compression which requires "
                f"additional codec support. Try the Recovery module, or contact support."
            )
        logger.info("[%s] Pixel data OK via [%s] strategy", result_id, decode_method)

        # ── Layer 2: Preprocess + get original image
        original_rgb = to_rgb(pixel_array, dicom_data=dicom_data)
        original_b64 = array_to_base64_png(original_rgb)

        # ── Layer 3: Multi-signal Forensic Detection
        logger.info("[%s] Running forensic detection", result_id)
        detection = classify(pixel_array)
        label = detection["label"]
        confidence = detection["confidence"]
        forensics = detection.get("forensics", {})

        # ── Layer 4: Grad-CAM Heatmap + Contour Overlay
        logger.info("[%s] Generating heatmap", result_id)
        hm_class = 2 if label == "AI-GENERATED" else 1
        heatmap_result  = generate_heatmap_full(pixel_array, class_idx=hm_class)
        heatmap_arr     = heatmap_result["heatmap"]
        overlay_arr     = heatmap_result["overlay"]
        heatmap_b64     = array_to_base64_png(heatmap_arr)
        overlay_b64     = array_to_base64_png(overlay_arr)

        heatmap_max = float(np.array(heatmap_result["cam_float"], dtype=np.float32).max())
        if label == "ORIGINAL" and forensics.get("ai_composite", 0) < 0.3:
            heatmap_max = min(heatmap_max, 0.4)

        # ── Layer 5: Metadata Forensic Validation
        logger.info("[%s] Validating metadata forensics", result_id)
        meta_validation  = validate_metadata(dicom_data["tags"], dicom_data["file_hash"])
        meta_suspicion   = meta_validation.get("suspicion_score", 0)

        # ── Layer 6: Full Trust Score with transparency breakdown
        trust_result = compute_trust_score_full(
            label=label,
            confidence=confidence,
            metadata_status=meta_validation["status"],
            heatmap_max=heatmap_max,
            forensics=forensics,
            metadata_suspicion_score=meta_suspicion,
        )
        trust = trust_result["trust_score"]

        # ── Build response
        response = {
            "result_id":   result_id,
            "filename":    file.filename,
            "label":       label,
            "confidence":  confidence,
            "probabilities": detection.get("probabilities", {}),
            # ─ Trust & reliability
            "trust_score":          trust,
            "confidence_tier":      trust_result["confidence_tier"],
            "clinical_reliability": trust_result["clinical_reliability"],
            "reliability_label":    trust_result["reliability_label"],
            "requires_review":      trust_result["requires_review"],
            # ─ Images (original + 3-mode heatmap)
            "original_image":  original_b64,
            "heatmap_image":   heatmap_b64,
            "overlay_image":   overlay_b64,
            # ─ Multi-signal fusion breakdown
            "fusion": trust_result["fusion"],
            # ─ Forensic signals
            "forensics": {
                "ai_composite":     forensics.get("ai_composite", 0),
                "tamper_composite": forensics.get("tamper_composite", 0),
                "fft_ai_score":     forensics.get("fft", {}).get("ai_score", 0),
                "noise_ai_score":   forensics.get("noise", {}).get("ai_score", 0),
                "texture_ai_score": forensics.get("texture", {}).get("ai_score", 0),
                "spectral_slope":   forensics.get("fft", {}).get("spectral_slope", 0),
                "noise_std":        forensics.get("noise", {}).get("noise_std", 0),
                "dct_uniform":      forensics.get("texture", {}).get("dct_uniform", False),
            },
            # ─ Evidence-based reasoning
            "forensic_reasoning": trust_result["forensic_reasoning"],
            "conclusion":          trust_result["conclusion"],
            # ─ Metadata
            "metadata": {
                "tags":          dicom_data["tags"],
                "present_count": dicom_data["present_count"],
                "total_count":   dicom_data["total_count"],
                "integrity": {
                    "status":          meta_validation["status"],
                    "issues":          meta_validation["issues"],
                    "warnings":        meta_validation.get("warnings", []),
                    "ai_indicators":   meta_validation.get("ai_indicators", []),
                    "suspicion_score": meta_suspicion,
                    "hash":            meta_validation["hash"],
                },
            },
        }

        _result_cache[result_id] = response
        _log_activity(file.filename, label, result_id)

        # ── Persist to local storage
        try:
            data_store.save_validation(
                case_id=result_id,
                filename=file.filename,
                label=label,
                trust_score=int(trust),
                confidence=float(confidence),
                image_b64=original_b64,
                heatmap_b64=heatmap_b64,
                extra={
                    "confidence_tier":      trust_result["confidence_tier"],
                    "clinical_reliability": trust_result["clinical_reliability"],
                    "requires_review":      trust_result["requires_review"],
                },
            )
        except Exception as _ds_err:
            logger.warning("[%s] data_store.save_validation failed: %s", result_id, _ds_err)

        logger.info("[%s] Done: %s | confidence=%.1f%% | trust=%d%% | tier=%s",
                    result_id, label, confidence, trust, trust_result["confidence_tier"])
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Validation error", result_id)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ─── Module 2: Recover ────────────────────────────────────────────────────────
@app.post("/recover", tags=["Module 2 — Recovery"])
async def recover_dicom(file: UploadFile = File(...)):
    """
    Detect corruption and recover a damaged DICOM file.

    Returns:
    - corrupted_image: base64 PNG (raw corrupted view)
    - recovered_image: base64 PNG (approximate reconstruction)
    - recovered_dicom_b64: base64 of recovered .dcm file
    - corruption_report: type, severity, affected_percentage
    - restored_metadata: dict of tag_name → value
    """
    if not (file.filename.lower().endswith(".dcm") or file.content_type in ("application/dicom", "application/octet-stream")):
        raise HTTPException(400, "Only .dcm DICOM files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")
    if len(file_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    result_id = str(uuid.uuid4())

    try:
        # ── Load DICOM (force=True for corrupted files)
        logger.info("[%s] Loading corrupted DICOM: %s", result_id, file.filename)
        dicom_data = load_dicom(file_bytes)
        pixel_array = dicom_data["pixel_array"]
        ds = dicom_data["ds"]

        # ── Layer 7: Corruption Detection
        logger.info("[%s] Detecting corruption", result_id)
        corruption = detect_corruption(pixel_array, ds)

        # Original corrupted image
        if pixel_array is not None:
            corrupted_rgb = to_rgb(pixel_array, dicom_data=dicom_data)
            corrupted_b64 = array_to_base64_png(corrupted_rgb)
        else:
            # All-black placeholder
            corrupted_b64 = array_to_base64_png(np.zeros((224, 224), dtype=np.uint8))

        # ── Layer 8: Recovery & Reconstruction
        logger.info("[%s] Recovering image", result_id)
        pixel_corruption_mask = corruption.get("mask", None)
        recovery_outputs = recover_image(pixel_array, pixel_corruption_mask)

        # ── Unpack all method outputs (recover_image now returns a dict)
        # Each value is a float array in original DICOM pixel range (or None for AI when unavailable)
        arr_original = recovery_outputs["original"]
        arr_opencv   = recovery_outputs["opencv"]
        arr_ai       = recovery_outputs["ai"]      # may be None
        arr_final    = recovery_outputs["final"]

        def _encode(arr):
            """Safely encode a numpy array (or None) → base64 PNG string."""
            if arr is None:
                return None
            return array_to_base64_png(to_rgb(arr, dicom_data=dicom_data))

        corrupted_b64 = array_to_base64_png(to_rgb(pixel_array, dicom_data=dicom_data)) \
                        if pixel_array is not None else \
                        array_to_base64_png(np.zeros((224, 224), dtype=np.uint8))

        recovered_b64       = _encode(arr_final)
        opencv_b64          = _encode(arr_opencv)
        ai_b64              = _encode(arr_ai)           # None if AE not applicable
        original_b64        = _encode(arr_original)

        # ── Layer 9: Metadata Restoration
        logger.info("[%s] Restoring metadata", result_id)
        ds_restored, restored_tags = restore_metadata(ds, dicom_data["tags"])

        # ── Write recovered DICOM to bytes
        try:
            recovered_buf = io.BytesIO()
            pydicom.dcmwrite(recovered_buf, ds_restored)
            recovered_dicom_b64 = base64.b64encode(recovered_buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning("[%s] Could not write recovered DICOM: %s", result_id, e)
            recovered_dicom_b64 = None

        # ── Build corruption report (remove numpy mask from JSON response)
        report_clean = {
            "type": corruption["type"],
            "severity": corruption["severity"],
            "affected_percentage": corruption["affected_percentage"],
            "recoverable": corruption["recoverable"],
            "description": corruption["description"],
            "metadata_issues": len(dicom_data["tags"]) - dicom_data["present_count"],
            "metadata_restored": len(restored_tags),
        }

        response = {
            "result_id":          result_id,
            "filename":           file.filename,
            "corrupted_image":    corrupted_b64,   # raw upload (backward compat)
            "recovered_image":    recovered_b64,   # final hybrid result (backward compat)
            "recovered_dicom_b64": recovered_dicom_b64,
            "corruption_report":  report_clean,
            "restored_metadata":  restored_tags,
            # ── Multi-method comparison images ──────────────────────────────
            "method_images": {
                "original": original_b64,   # normalized corrupted input
                "opencv":   opencv_b64,     # Traditional: NLMeans+Bilateral+Unsharp
                "ai":       ai_b64,         # AI-only: autoencoder (None if unavailable)
                "final":    recovered_b64,  # Hybrid: full pipeline result
            },
            "ai_available": ai_b64 is not None,
        }

        _result_cache[result_id] = response
        _log_activity(file.filename, "RECOVERED", result_id)

        # ── Persist to local storage
        try:
            _affected = corruption.get("affected_percentage", 0.0)
            _quality  = max(0.0, round(1.0 - (_affected / 100.0), 4)) if _affected else None
            data_store.save_recovery(
                case_id=result_id,
                filename=file.filename,
                recovered=corruption.get("recoverable", True),
                corruption_type=corruption.get("type", "unknown"),
                severity=corruption.get("severity", "unknown"),
                affected_pct=float(_affected),
                quality_score=_quality,
                image_b64=corrupted_b64,
                recovered_image_b64=recovered_b64,
            )
        except Exception as _ds_err:
            logger.warning("[%s] data_store.save_recovery failed: %s", result_id, _ds_err)

        logger.info("[%s] Recovery complete: %s | severity=%s | metadata_restored=%d | ai_available=%s",
                    result_id, corruption["type"], corruption["severity"], len(restored_tags), ai_b64 is not None)
        return JSONResponse(content=response)


    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Recovery error", result_id)
        raise HTTPException(500, f"Recovery failed: {str(e)}")


# ─── Module 3: Radiology Report ──────────────────────────────────────────────
@app.post("/radiology-report", tags=["Module 3 — Radiology Report"])
async def radiology_report(file: UploadFile = File(...)):
    """
    Generate a structured AI clinical radiology report from a DICOM file.

    Calls Gemini Vision to produce a detailed structured report including:
    - Clinical indication, technique, findings per anatomical region
    - Impression bullets, recommendation
    - Integrity note based on forensic classification

    Returns:
    - report: structured JSON report
    - label, confidence, trust_score: reused from inline validation
    - generated_by: model that produced the report
    """
    if not (file.filename.lower().endswith(".dcm") or file.content_type in ("application/dicom", "application/octet-stream")):
        raise HTTPException(400, "Only .dcm DICOM files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")
    if len(file_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    result_id = str(uuid.uuid4())
    try:
        # Re-run validation inline to get pixel + metadata
        dicom_data = load_dicom(file_bytes)
        pixel_array = dicom_data["pixel_array"]
        tags = dicom_data["tags"]

        if pixel_array is None:
            raise HTTPException(422, "Could not decode pixel data for radiology report generation.")

        # Detection for label, confidence, trust
        from preprocessor import to_rgb
        rgb_array = to_rgb(pixel_array, dicom_data=dicom_data)

        detection = classify(pixel_array)
        label = detection["label"]
        confidence = detection["confidence"]
        forensics = detection.get("forensics", {})

        hm_class = 2 if label == "AI-GENERATED" else 1
        heatmap_result = generate_heatmap_full(pixel_array, class_idx=hm_class)
        heatmap_max = float(np.array(heatmap_result["cam_float"], dtype=np.float32).max())
        if label == "ORIGINAL" and forensics.get("ai_composite", 0) < 0.3:
            heatmap_max = min(heatmap_max, 0.4)

        meta_validation = validate_metadata(tags, dicom_data["file_hash"])
        trust_result = compute_trust_score_full(
            label=label,
            confidence=confidence,
            metadata_status=meta_validation["status"],
            heatmap_max=heatmap_max,
            forensics=forensics,
            metadata_suspicion_score=meta_validation.get("suspicion_score", 0),
        )
        trust = trust_result["trust_score"]

        # Generate clinical radiology report
        logger.info("[%s] Generating radiology report for: %s", result_id, file.filename)
        report = generate_radiology_report(
            rgb_array=rgb_array,
            tags=tags,
            label=label,
            trust_score=trust,
            confidence=confidence,
        )

        _log_activity(file.filename, "REPORT", result_id)
        return JSONResponse(content={
            "result_id": result_id,
            "filename": file.filename,
            "label": label,
            "confidence": confidence,
            "trust_score": trust,
            "confidence_tier": trust_result["confidence_tier"],
            "report": report,
            "generated_by": report.get("generated_by", "unknown"),
        })

    except RateLimitError as e:
        logger.warning("[%s] Gemini rate limited: %s", result_id, e)
        raise HTTPException(
            429,
            detail="Gemini API rate limit reached. The free tier allows 15 requests per minute. "
                   "Please wait 60 seconds and try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Radiology report error", result_id)
        raise HTTPException(500, f"Radiology report generation failed: {str(e)}")


# ─── Activity Feed ────────────────────────────────────────────────────────────
@app.get("/activity", tags=["System"])
async def get_activity(limit: int = 10):
    """Return the most recent scan/recovery/report activity entries (newest first)."""
    limit = max(1, min(limit, MAX_ACTIVITY))
    return JSONResponse(content={"activity": list(reversed(_activity_log))[:limit]})


# ─── Cases Feed (persistent, disk-backed) ─────────────────────────────────────
@app.get("/cases", tags=["Cases"])
async def get_cases(limit: int = 50, module: str = ""):
    """
    Return merged validation + recovery case records (newest first).

    Query params
    ------------
    limit  : max number of cases to return (1-200, default 50)
    module : filter to 'validation' | 'recovery' | '' (all)
    """
    limit = max(1, min(limit, 200))
    cases = data_store.load_cases()

    if module == "validation":
        cases = [c for c in cases if c.get("has_validation")]
    elif module == "recovery":
        cases = [c for c in cases if c.get("has_recovery")]

    return JSONResponse(content={"cases": cases[:limit], "total": len(cases)})


# ─── Serve saved images ───────────────────────────────────────────────────────
from fastapi.responses import FileResponse as _FileResponse

@app.get("/images/{sub_path:path}", tags=["Cases"])
async def serve_image(sub_path: str):
    """
    Serve a PNG from the data/ directory.
    e.g. GET /images/validation/images/case_001_orig.png
    """
    full_path = data_store.DATA_ROOT / sub_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, f"Image not found: {sub_path}")
    # Prevent path traversal
    try:
        full_path.resolve().relative_to(data_store.DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    return _FileResponse(str(full_path), media_type="image/png")


# ─── Platform Stats (real data from disk) ────────────────────────────────────
@app.get("/stats", tags=["System"])
async def get_stats():
    """
    Return real-time platform statistics derived from persisted case records.

    Response fields
    ---------------
    scans_processed : total validation cases stored on disk
    threats_detected: validation cases labelled TAMPERED or AI-GENERATED
    files_recovered : recovery cases where recovered == True
    detection_accuracy: fixed model accuracy (80.4 %)
    """
    val_records = data_store.load_validation()
    rec_records = data_store.load_recovery()

    scans     = len(val_records)
    threats   = sum(1 for v in val_records if v.get("label", "") in ("TAMPERED", "AI-GENERATED"))
    recovered = sum(1 for r in rec_records if r.get("recovered") is True)

    return JSONResponse(content={
        "scans_processed":    scans,
        "threats_detected":   threats,
        "files_recovered":    recovered,
        "detection_accuracy": 80.4,
    })


# ─── Fetch result by ID ───────────────────────────────────────────────────────
@app.get("/result/{result_id}", tags=["System"])
async def get_result(result_id: str):
    """Retrieve a previously computed result by its ID."""
    result = _result_cache.get(result_id)
    if not result:
        raise HTTPException(404, f"Result '{result_id}' not found or expired")
    return JSONResponse(content=result)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
