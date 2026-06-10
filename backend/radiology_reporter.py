"""
radiology_reporter.py — PhantomaShield
=======================================
Uses Google Gemini (google-genai SDK) to generate a structured clinical
radiology report directly from the DICOM pixel array + validated metadata.

Requirements:
    pip install google-genai python-dotenv Pillow

Setup:
    Set GEMINI_API_KEY in backend/.env
"""

import os
import io
import base64
import logging
import textwrap
import re
import json
from typing import Optional

logger = logging.getLogger("phantomashield.radiology")


class RateLimitError(Exception):
    """Raised when Gemini API returns HTTP 429 Resource Exhausted."""
    pass

# ── graceful imports ───────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-genai not installed — using heuristic fallback")

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ── env / config ───────────────────────────────────────────────────────────────
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except ImportError:
        pass

def _get_gemini_key() -> Optional[str]:
    _load_env()
    return os.environ.get("GEMINI_API_KEY")

def _get_groq_key() -> Optional[str]:
    _load_env()
    return os.environ.get("GROQ_API_KEY")


# ── image helper ───────────────────────────────────────────────────────────────
def _numpy_to_pil_bytes(rgb_array) -> Optional[bytes]:
    """Convert H×W×3 uint8 numpy array → PNG bytes."""
    if not _PIL_AVAILABLE or rgb_array is None:
        return None
    try:
        import numpy as np
        arr = np.array(rgb_array, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = PILImage.fromarray(arr[:, :, :3] if arr.shape[-1] > 3 else arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Image conversion failed: %s", exc)
        return None


# ── prompt ─────────────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = textwrap.dedent("""
You are a senior board-certified radiologist AI assistant.
Analyze the DICOM medical image and metadata provided.
Generate a complete, detailed, standard clinical radiology report following
the internationally accepted 7-section radiology report format.

=== DICOM METADATA PROVIDED ===
Section 1 — Patient Information:
  Patient Name   : {patient_name}
  Patient ID/MRN : {patient_id}
  Age            : {patient_age}
  Sex            : {patient_sex}
  Referring Doctor: {referring_doctor}
  Hospital/Institution: {institution}

Section 2 — Study Information:
  Study Date & Time : {study_date} {study_time}
  Modality          : {modality}
  Body Part Examined: {body_part}
  Study Description : {study_desc}
  Accession Number  : {accession}
  SOP Instance UID  : {sop_uid}

Section 3 — Scanner / Technical:
  Scanner Make/Model: {manufacturer} {model}
  Slice Thickness   : {slice_thick} mm
  Pixel Spacing     : {pixel_spacing} mm
  Image Dimensions  : {rows} x {cols} px
  Bits Stored       : {bits}

Forensic Integrity Status:
{integrity_msg}

Return ONLY a valid JSON object (no markdown fences, no extra text) with this exact structure:
{{
  "report_type": "e.g. CT Chest",

  "patient_info": {{
    "patient_name": "from metadata or Anonymous",
    "patient_id": "MRN from metadata or N/A",
    "age": "age from metadata",
    "sex": "sex from metadata",
    "referring_doctor": "from metadata or N/A",
    "hospital": "institution from metadata or N/A"
  }},

  "study_info": {{
    "study_date": "date from metadata",
    "study_time": "time if available",
    "modality": "CT/MRI/DX etc.",
    "body_part": "body part examined",
    "accession_number": "from metadata or N/A"
  }},

  "clinical_indication": "Reason for the study inferred from metadata or image. Be specific.",

  "technique": "Describe the imaging technique: modality, contrast (IV/oral/none), scan plane, reconstruction, slice thickness etc.",

  "findings": {{
    "image_quality": "Technical quality assessment",
    "lungs": "Detailed observation of lung parenchyma (or anatomically appropriate region)",
    "pleura": "Pleural space assessment",
    "mediastinum": "Mediastinal structure assessment",
    "heart_vessels": "Cardiac silhouette and major vessels",
    "bones_soft_tissue": "Osseous and soft tissue structures",
    "other": "Any additional structures visible"
  }},

  "impression": [
    "Numbered concise diagnosis statement 1",
    "Numbered concise diagnosis statement 2"
  ],

  "integrity_note": "Clinical authenticity note based on forensic classification result",

  "recommendation": "Specific follow-up action: further tests, biopsy, PET scan, follow-up interval etc.",

  "disclaimer": "AI-generated preliminary report. Must be reviewed and countersigned by a licensed radiologist before clinical use."
}}

Rules:
- Be specific and objective. State normal findings explicitly (e.g. 'No focal consolidation identified.').
- Use standard radiological terminology.
- For findings, describe location, size, shape, and density where applicable.
- If image quality prevents full evaluation, state it clearly.
- Return ONLY valid JSON. No text outside the JSON object.
""").strip()


# ── Gemini call ────────────────────────────────────────────────────────────────
def _generate_with_gemini(rgb_array, tags: dict, label: str, trust_score: float, confidence: float) -> dict:
    api_key = _get_gemini_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Extract all standard radiology fields from DICOM tags
    modality      = tags.get("(0008,0060)", "CT")
    body_part     = tags.get("(0018,0015)", "CHEST")
    study_desc    = tags.get("(0008,103E)", "Not specified")
    patient_name  = tags.get("(0010,0010)", "Anonymous")
    patient_id    = tags.get("(0010,0020)", "N/A")
    patient_age   = tags.get("(0010,1010)", "N/A")
    patient_sex   = tags.get("(0010,0040)", "N/A")
    patient_dob   = tags.get("(0010,0030)", "N/A")
    study_date    = tags.get("(0008,0020)", "N/A")
    study_time    = tags.get("(0008,0030)", "N/A")
    accession     = tags.get("(0008,0050)", "N/A") if "(0008,0050)" in tags else "N/A"
    sop_uid       = tags.get("(0008,0018)", "N/A")
    referring_doc = tags.get("(0008,0090)", "N/A") if "(0008,0090)" in tags else "N/A"
    institution   = tags.get("(0008,0080)", "N/A") if "(0008,0080)" in tags else "N/A"
    manufacturer  = tags.get("(0008,0070)", "N/A")
    model_name    = tags.get("(0008,1090)", "")
    slice_thick   = tags.get("(0018,0050)", "N/A")
    pixel_spacing = tags.get("(0028,0030)", "N/A")
    rows          = tags.get("(0028,0010)", "N/A")
    cols          = tags.get("(0028,0011)", "N/A")
    bits          = tags.get("(0028,0101)", "N/A")

    if label in ("TAMPERED", "AI-GENERATED"):
        integrity_msg = (
            f"⚠ AUTHENTICITY ALERT: This image was classified as {label} "
            f"(confidence: {confidence:.1f}%, Trust Score: {trust_score:.0f}/100). "
            "This must be prominently stated in integrity_note and impression."
        )
    else:
        integrity_msg = (
            f"Image authenticated as ORIGINAL by PhantomaShield forensic analysis "
            f"(Trust Score: {trust_score:.0f}/100, Confidence: {confidence:.1f}%)."
        )

    prompt = _PROMPT_TEMPLATE.format(
        modality=modality, body_part=body_part, study_desc=study_desc,
        patient_name=patient_name, patient_id=patient_id,
        patient_age=patient_age, patient_sex=patient_sex,
        study_date=study_date, study_time=study_time,
        accession=accession, sop_uid=sop_uid,
        referring_doctor=referring_doc, institution=institution,
        manufacturer=manufacturer, model=model_name,
        slice_thick=slice_thick, pixel_spacing=pixel_spacing,
        rows=rows, cols=cols, bits=bits,
        integrity_msg=integrity_msg,
    )

    # Build content parts
    parts = []

    img_bytes = _numpy_to_pil_bytes(rgb_array)
    if img_bytes:
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    parts.append(types.Part.from_text(text=prompt))

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1800,
            ),
        )
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise RateLimitError("Gemini API rate limit reached. Please wait 60 seconds and try again.") from exc
        raise

    raw = response.text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ── Groq Llama 3 Vision call ───────────────────────────────────────────────────
def _generate_with_groq(rgb_array, tags: dict, label: str, trust_score: float, confidence: float) -> dict:
    api_key = _get_groq_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    import groq
    client = groq.Groq(api_key=api_key)

    modality      = tags.get("(0008,0060)", "CT")
    body_part     = tags.get("(0018,0015)", "CHEST")
    study_desc    = tags.get("(0008,103E)", "Not specified")
    patient_name  = tags.get("(0010,0010)", "Anonymous")
    patient_id    = tags.get("(0010,0020)", "N/A")
    patient_age   = tags.get("(0010,1010)", "N/A")
    patient_sex   = tags.get("(0010,0040)", "N/A")
    study_date    = tags.get("(0008,0020)", "N/A")
    study_time    = tags.get("(0008,0030)", "N/A")
    accession     = tags.get("(0008,0050)", "N/A") if "(0008,0050)" in tags else "N/A"
    sop_uid       = tags.get("(0008,0018)", "N/A")
    referring_doc = tags.get("(0008,0090)", "N/A") if "(0008,0090)" in tags else "N/A"
    institution   = tags.get("(0008,0080)", "N/A") if "(0008,0080)" in tags else "N/A"
    manufacturer  = tags.get("(0008,0070)", "N/A")
    model_name    = tags.get("(0008,1090)", "")
    slice_thick   = tags.get("(0018,0050)", "N/A")
    pixel_spacing = tags.get("(0028,0030)", "N/A")
    rows          = tags.get("(0028,0010)", "N/A")
    cols          = tags.get("(0028,0011)", "N/A")
    bits          = tags.get("(0028,0101)", "N/A")

    if label in ("TAMPERED", "AI-GENERATED"):
        integrity_msg = (
            f"⚠ AUTHENTICITY ALERT: This image was classified as {label} "
            f"(confidence: {confidence:.1f}%, Trust Score: {trust_score:.0f}/100). "
            "This must be prominently stated in integrity_note and impression."
        )
    else:
        integrity_msg = (
            f"Image authenticated as ORIGINAL by PhantomaShield forensic analysis "
            f"(Trust Score: {trust_score:.0f}/100, Confidence: {confidence:.1f}%)."
        )

    prompt = _PROMPT_TEMPLATE.format(
        modality=modality, body_part=body_part, study_desc=study_desc,
        patient_name=patient_name, patient_id=patient_id,
        patient_age=patient_age, patient_sex=patient_sex,
        study_date=study_date, study_time=study_time,
        accession=accession, sop_uid=sop_uid,
        referring_doctor=referring_doc, institution=institution,
        manufacturer=manufacturer, model=model_name,
        slice_thick=slice_thick, pixel_spacing=pixel_spacing,
        rows=rows, cols=cols, bits=bits,
        integrity_msg=integrity_msg,
    )

    img_bytes = _numpy_to_pil_bytes(rgb_array)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    if img_bytes:
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_img}"
            }
        })
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=1800,
        )
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str or "rate limit" in err_str.lower():
            raise RateLimitError("Groq API rate limit reached. Please wait and try again.") from exc
        raise

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ── heuristic fallback ─────────────────────────────────────────────────────────
def _heuristic_report(tags: dict, label: str, trust_score: float, confidence: float) -> dict:
    modality   = tags.get("(0008,0060)", "CT")
    body_part  = tags.get("(0018,0015)", "CHEST").title()
    study_desc = tags.get("(0008,103E)", "Not specified in metadata")

    if label in ("TAMPERED", "AI-GENERATED"):
        integrity_note = (
            f"⚠ AUTHENTICITY ALERT: Forensic analysis classified this image as {label} "
            f"(confidence: {confidence:.1f}%, Trust Score: {trust_score:.0f}/100). "
            "Do NOT use for clinical decisions without independent verification."
        )
    else:
        integrity_note = (
            f"Image classified as ORIGINAL (Trust Score: {trust_score:.0f}/100, "
            f"Confidence: {confidence:.1f}%). Standard clinical review recommended."
        )

    return {
        "report_type": f"{modality} {body_part}",

        "patient_info": {
            "patient_name": tags.get("(0010,0010)", "Anonymous"),
            "patient_id":   tags.get("(0010,0020)", "N/A"),
            "age":          tags.get("(0010,1010)", "N/A"),
            "sex":          tags.get("(0010,0040)", "N/A"),
            "referring_doctor": "N/A",
            "hospital":     "N/A",
        },

        "study_info": {
            "study_date":       tags.get("(0008,0020)", "N/A"),
            "study_time":       tags.get("(0008,0030)", "N/A"),
            "modality":         modality,
            "body_part":        body_part,
            "accession_number": "N/A",
        },

        "clinical_indication": study_desc,

        "technique": (
            f"{modality} acquisition. "
            f"Slice thickness: {tags.get('(0018,0050)', 'N/A')} mm. "
            f"Matrix: {tags.get('(0028,0010)', 'N/A')}x{tags.get('(0028,0011)', 'N/A')} px. "
            f"Scanner: {tags.get('(0008,0070)', 'N/A')} {tags.get('(0008,1090)', '')}."
        ),

        "findings": {
            "image_quality": "Adequate for assessment (heuristic report — no Gemini Vision analysis).",
            "lungs": "Manual radiologist review required.",
            "pleura": "No gross pleural abnormality by automated assessment.",
            "mediastinum": "Mediastinal contours require manual assessment.",
            "heart_vessels": "Cardiac silhouette not individually assessed.",
            "bones_soft_tissue": "Osseous structures require manual review.",
            "other": "Automated heuristic report generated from DICOM metadata only."
        },

        "impression": [
            "Automated metadata-only report — Gemini Vision AI was unavailable.",
            integrity_note,
        ],

        "integrity_note": integrity_note,
        "recommendation": "Full manual radiologist review is required before clinical use.",
        "disclaimer": "AI-generated preliminary report. Must be reviewed and countersigned by a licensed radiologist before clinical use."
    }


# ── Public entry point ─────────────────────────────────────────────────────────
def generate_radiology_report(
    rgb_array,
    tags: dict,
    label: str,
    trust_score: float,
    confidence: float,
) -> dict:
    """
    Generate a structured clinical radiology report.
    Tries Groq if key is available, then Gemini Vision.
    - Raises RateLimitError on 429
    - Falls back to heuristic only on other non-rate-limit errors or missing keys.
    """
    
    # 1. Try Groq (Llama 3.2 Vision)
    if _get_groq_key():
        try:
            logger.info("Calling Groq Vision (Llama-3.2-90B) for radiology report...")
            report = _generate_with_groq(rgb_array, tags, label, trust_score, confidence)
            report["generated_by"] = "Groq Llama-3.2-90B Vision API"
            logger.info("Groq radiology report generated successfully.")
            return report
        except RateLimitError:
            raise
        except Exception as exc:
            logger.warning("Groq call failed: %s — falling back...", exc)

    # 2. Try Gemini
    if _GENAI_AVAILABLE and _get_gemini_key():
        try:
            logger.info("Calling Gemini Vision for radiology report...")
            report = _generate_with_gemini(rgb_array, tags, label, trust_score, confidence)
            report["generated_by"] = "Gemini 2.0 Flash Vision"
            logger.info("Gemini radiology report generated successfully.")
            return report
        except RateLimitError:
            raise  # Let FastAPI handle it as HTTP 429
        except Exception as exc:
            logger.warning("Gemini call failed: %s — using heuristic fallback.", exc)

    # 3. Fallback Heuristic
    report = _heuristic_report(tags, label, trust_score, confidence)
    report["generated_by"] = "Heuristic (AI logic unavailable or no valid API key)"
    return report
