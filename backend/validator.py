"""
validator.py
DICOM metadata forensic validator.
Checks for:
  - Missing required tags
  - "Suspiciously perfect" metadata (AI-generated DICOMs often have all tags present)
  - Tag value inconsistencies (date mismatches, UID format errors)
  - Known AI-generation software markers
  - Implausible field values
  - Absence of scanner-specific private tags (real scanners always add these)

Status:
  VALID     → authentic-looking metadata
  SUSPICIOUS → possibly synthetic or AI-generated metadata
  MODIFIED  → clearly missing required tags or invalid values
"""
import re
import hashlib
from dicom_loader import CRITICAL_TAGS


REQUIRED_TAGS = {
    "(0008,0060)",   # Modality
    "(0010,0010)",   # PatientName
    "(0010,0020)",   # PatientID
    "(0008,0016)",   # SOPClassUID
    "(0008,0018)",   # SOPInstanceUID
    "(0028,0010)",   # Rows
    "(0028,0011)",   # Columns
    "(0028,0100)",   # BitsAllocated
}

VALID_MODALITIES = {
    "CT", "MR", "MRI", "US", "XR", "DX", "CR", "PT", "NM",
    "OT", "SC", "RF", "MG", "IO", "PX", "GM", "SM", "XC",
}

VALID_BITS = {"8", "12", "16", "32"}

# Known AI/synthetic generation software signatures
AI_SOFTWARE_MARKERS = [
    "synthetic", "phantom", "ai", "artificial", "generated",
    "fake", "simulated", "virtual", "dcmtk", "cornerstonejs",
    "dcm2niix", "plastimatch", "pydicom_gen", "test",
]

# Generic/default patient names that suggest synthetic data
GENERIC_PATIENT_NAMES = [
    "anonymous", "unknown", "test", "patient", "demo", "example",
    "sample", "fake", "phantom", "synthetic", "dummy",
]

# Implausibly generic UIDs (sequential or too-short UIDs may indicate synthetic)
SYNTHETIC_UID_PATTERNS = [
    r"^1\.2\.3\.",        # Very short root (not from a real DICOM vendor)
    r"^0\.",              # Leading zero root
    r"\.0+$",             # Trailing zeros
    r"^1\.2\.840\.10008\.", # SOPClassUID used as InstanceUID (wrong context)
]


def validate_metadata(tags: dict, file_hash: str = None) -> dict:
    """
    Forensic metadata validation.
    Returns:
      {
        status: "VALID" | "SUSPICIOUS" | "MODIFIED",
        issues: [...],          # Critical problems
        warnings: [...],        # Suspicious indicators
        ai_indicators: [...],   # Specific AI-generation flags
        suspicion_score: 0-100, # How suspicious overall
        hash: str,
        present_count: int,
        total_count: int,
      }
    """
    issues = []
    warnings = []
    ai_indicators = []
    suspicion_score = 0

    present = sum(1 for v in tags.values() if v is not None)
    total = len(tags)

    # ── 1. Required tag check ─────────────────────────────────────────────────
    for tag in REQUIRED_TAGS:
        if tags.get(tag) is None:
            issues.append(f"Missing required tag {tag} ({CRITICAL_TAGS.get(tag, '?')})")

    # ── 2. Suspiciously COMPLETE metadata ─────────────────────────────────────
    # Real-world DICOM files almost always have some missing optional tags.
    # AI-generated files often have EVERY tag filled with a plausible value.
    completeness_ratio = present / total if total > 0 else 0
    if completeness_ratio >= 0.95 and total >= 20:
        ai_indicators.append(
            f"Metadata completeness {present}/{total} ({completeness_ratio*100:.0f}%) is "
            f"unusually high — real scanners often omit optional tags"
        )
        suspicion_score += 20

    # ── 3. Modality format check ──────────────────────────────────────────────
    modality = tags.get("(0008,0060)")
    if modality:
        if modality.upper() not in VALID_MODALITIES:
            warnings.append(f"Unusual modality: '{modality}'")
            suspicion_score += 10

    # ── 4. Patient identity analysis ──────────────────────────────────────────
    patient_name = tags.get("(0010,0010)", "")
    if patient_name:
        name_lower = str(patient_name).lower().replace("^", " ").replace("_", " ")
        for generic in GENERIC_PATIENT_NAMES:
            if generic in name_lower:
                ai_indicators.append(
                    f"Patient name '{patient_name}' matches generic/synthetic placeholder"
                )
                suspicion_score += 25
                break

    patient_id = tags.get("(0010,0020)", "")
    if patient_id:
        pid = str(patient_id).strip()
        if pid.upper() in ("UNKNOWN", "NONE", "TEST", "DEMO", "SYNTHETIC", "0", "1", "123"):
            ai_indicators.append(f"Patient ID '{pid}' is a generic placeholder value")
            suspicion_score += 20
        if re.match(r"^[0-9]+$", pid) and len(pid) <= 3:
            warnings.append(f"Suspiciously short numeric Patient ID: '{pid}'")
            suspicion_score += 10

    # ── 5. UID forensics ──────────────────────────────────────────────────────
    for uid_tag, uid_name in [("(0008,0016)", "SOPClassUID"), ("(0008,0018)", "SOPInstanceUID"),
                               ("(0020,000D)", "StudyInstanceUID"), ("(0020,000E)", "SeriesInstanceUID")]:
        uid = tags.get(uid_tag)
        if uid:
            # Format check
            if not re.match(r"^[\d.]+$", str(uid)):
                warnings.append(f"Non-standard {uid_name}: '{str(uid)[:40]}'")
                suspicion_score += 8
            # Known synthetic patterns
            for pattern in SYNTHETIC_UID_PATTERNS:
                if re.search(pattern, str(uid)):
                    ai_indicators.append(f"{uid_name} matches synthetic UID pattern: '{str(uid)[:40]}'")
                    suspicion_score += 15
                    break

    # Check that SOPInstanceUID is different from StudyInstanceUID (should always be different)
    sop_instance = tags.get("(0008,0018)")
    study_instance = tags.get("(0020,000D)")
    if sop_instance and study_instance and str(sop_instance).strip() == str(study_instance).strip():
        ai_indicators.append("SOPInstanceUID == StudyInstanceUID — impossible in real DICOM")
        suspicion_score += 30
        issues.append("SOPInstanceUID and StudyInstanceUID are identical (invalid)")

    # ── 6. Date/time consistency ──────────────────────────────────────────────
    study_date = tags.get("(0008,0020)")
    if study_date:
        clean_date = str(study_date).replace("-", "").replace("/", "").strip()
        if not re.match(r"^\d{8}$", clean_date):
            warnings.append(f"Non-standard StudyDate format: '{study_date}'")
            suspicion_score += 8
        else:
            # Check for obviously fake dates
            year = int(clean_date[:4])
            if year < 1980 or year > 2030:
                ai_indicators.append(f"Implausible StudyDate year: {year}")
                suspicion_score += 20
            # Exact round dates (e.g. 20200101) are common in synthetic data
            if clean_date[4:] in ("0101", "0100", "0000", "1231"):
                warnings.append(f"StudyDate {study_date} is a round/default date (common in synthetic data)")
                suspicion_score += 10

    # ── 7. Bits allocated / pixel format consistency ──────────────────────────
    bits = tags.get("(0028,0100)")
    bits_stored = tags.get("(0028,0101)")
    if bits and bits not in VALID_BITS:
        warnings.append(f"Non-standard BitsAllocated: '{bits}'")
        suspicion_score += 8
    if bits and bits_stored:
        try:
            if int(str(bits_stored)) > int(str(bits)):
                issues.append(f"BitsStored ({bits_stored}) > BitsAllocated ({bits}) — impossible")
                suspicion_score += 30
        except ValueError:
            pass

    # ── 8. Manufacturer check ─────────────────────────────────────────────────
    manufacturer = tags.get("(0008,0070)", "")
    if manufacturer:
        mfr_lower = str(manufacturer).lower()
        for marker in AI_SOFTWARE_MARKERS:
            if marker in mfr_lower:
                ai_indicators.append(f"Manufacturer field contains AI/synthetic marker: '{manufacturer}'")
                suspicion_score += 35
                break
        # Check for generic/missing manufacturer (real scanners always have this)
        if mfr_lower.strip() in ("unknown", "none", "", "test"):
            warnings.append(f"Manufacturer is generic/missing: '{manufacturer}'")
            suspicion_score += 12

    # ── 9. Rows/Columns plausibility ─────────────────────────────────────────
    rows = tags.get("(0028,0010)")
    cols = tags.get("(0028,0011)")
    if rows and cols:
        try:
            r, c = int(str(rows)), int(str(cols))
            # Real medical images are typically 64×64 to 4096×4096
            if r < 32 or c < 32:
                warnings.append(f"Unusually small image size: {r}×{c}")
                suspicion_score += 5
            # Perfectly square power-of-2 sizes are common in synthetic data
            if r == c and r in (64, 128, 224, 256, 512):
                warnings.append(
                    f"Image size {r}×{c} is a common ML/AI power-of-2 size — may indicate synthetic origin"
                )
                suspicion_score += 15
        except ValueError:
            pass

    # ── 10. Determine final status ────────────────────────────────────────────
    suspicion_score = min(100, suspicion_score)

    if len(issues) > 0:
        status = "MODIFIED"
    elif suspicion_score >= 35 or len(ai_indicators) >= 2:
        status = "SUSPICIOUS"
    elif suspicion_score >= 15 or len(ai_indicators) >= 1:
        status = "SUSPICIOUS"
    else:
        status = "VALID"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "ai_indicators": ai_indicators,
        "suspicion_score": suspicion_score,
        "hash": file_hash,
        "present_count": present,
        "total_count": total,
    }
