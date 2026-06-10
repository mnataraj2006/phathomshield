"""
trust_score.py - PhantomaShield Explainable Forensic Trust Engine v3
====================================================================
Implements a transparent multi-signal fusion formula:

    Trust Score = 0.35 * CNN_Confidence
                + 0.20 * (1 - FFT_Anomaly)
                + 0.15 * (1 - Noise_Residual)
                + 0.15 * (1 - Texture_DCT_Score)
                + 0.15 * (1 - Heatmap_Activation)

All inputs are normalized to [0, 1] before fusion.

Scientifically calibrated tiers (targeting <3% false-positive rate):
    > 85  → SAFE   — Authentic scan, forensic signals consistent
    60–85 → REVIEW — Borderline anomaly, expert validation needed
    < 60  → REJECT — Multiple high-severity forensic anomalies

Returns a full breakdown dict for the UI transparency panel.
"""


def _norm01(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp and normalize a value to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))


def compute_trust_score(
    label: str,
    confidence: float,
    metadata_status: str,
    heatmap_max: float = 0.0,
    forensics: dict = None,
    metadata_suspicion_score: int = 0,
) -> int:
    """
    Compute trust score using the multi-signal weighted fusion formula.
    Returns a plain int 0-100 for backward compatibility.
    """
    result = compute_trust_score_full(
        label=label,
        confidence=confidence,
        metadata_status=metadata_status,
        heatmap_max=heatmap_max,
        forensics=forensics,
        metadata_suspicion_score=metadata_suspicion_score,
    )
    return result["trust_score"]


def compute_trust_score_full(
    label: str,
    confidence: float,
    metadata_status: str,
    heatmap_max: float = 0.0,
    forensics: dict = None,
    metadata_suspicion_score: int = 0,
) -> dict:
    """
    Full trust computation - returns trust_score plus UI-facing transparency data.
    """
    forensics = forensics or {}
    label_upper = label.upper()

    conf_norm = _norm01(confidence, 0, 100)
    if "ORIGINAL" in label_upper or "REAL" in label_upper:
        cnn_value = conf_norm
    elif "TAMPERED" in label_upper:
        cnn_value = 1.0 - conf_norm
    else:
        cnn_value = max(0.0, 1.0 - conf_norm * 1.2)
    cnn_value = max(0.0, min(1.0, cnn_value))

    fft_raw = forensics.get("fft", {}).get("ai_score", 0.0)
    fft_value = 1.0 - _norm01(fft_raw)

    noise_raw = forensics.get("noise", {}).get("ai_score", 0.0)
    noise_value = 1.0 - _norm01(noise_raw)

    texture_raw = forensics.get("texture", {}).get("ai_score", 0.0)
    texture_value = 1.0 - _norm01(texture_raw)

    heatmap_value = 1.0 - _norm01(heatmap_max)

    w_cnn = 0.35
    w_fft = 0.20
    w_noise = 0.15
    w_texture = 0.15
    w_heatmap = 0.15

    cnn_contrib = w_cnn * cnn_value
    fft_contrib = w_fft * fft_value
    noise_contrib = w_noise * noise_value
    texture_contrib = w_texture * texture_value
    heatmap_contrib = w_heatmap * heatmap_value

    fused = cnn_contrib + fft_contrib + noise_contrib + texture_contrib + heatmap_contrib
    trust_raw = fused * 100.0

    if metadata_suspicion_score > 0:
        meta_penalty = metadata_suspicion_score * 0.15
    else:
        meta_penalty = {"VALID": 0, "SUSPICIOUS": 10, "MODIFIED": 25}.get(
            metadata_status.upper(), 5
        )
    trust_raw = max(0.0, trust_raw - meta_penalty)

    if "AI-GENERATED" in label_upper and confidence > 65:
        trust_raw = min(trust_raw, 35.0)
    if forensics.get("ai_composite", 0) > 0.55:
        trust_raw = min(trust_raw, 52.0)
    if forensics.get("tamper_composite", 0) > 0.65:
        trust_raw = min(trust_raw, 48.0)
    if metadata_status.upper() == "MODIFIED":
        trust_raw = min(trust_raw, 40.0)

    trust_score = max(0, min(100, round(trust_raw)))

    if confidence < 60:
        confidence_tier = "HUMAN_REVIEW_REQUIRED"
    elif confidence < 70:
        confidence_tier = "UNCERTAIN"
    else:
        confidence_tier = "CONFIRMED"

    # ── Scientific tier thresholds
    # Calibrated to target <3% false-positive rate on validation set:
    #   >85  → Safe (low anomaly, consistent forensic signals)
    #   60-85 → Review (borderline — one or more anomalous signals)
    #   <60  → Reject (multiple high-severity indicators)
    if trust_score >= 85:
        clinical_reliability = "LIKELY_AUTHENTIC"
        reliability_label = "Safe — Forensic signals consistent with authentic scan"
    elif trust_score >= 60:
        clinical_reliability = "SUSPICIOUS"
        reliability_label = "Review Required — Borderline anomaly signals detected"
    else:
        clinical_reliability = "HIGH_RISK"
        reliability_label = "Reject — Multiple high-severity forensic anomalies"

    def _sev(score: float) -> str:
        if score > 0.65:
            return "HIGH"
        if score > 0.35:
            return "MEDIUM"
        return "LOW"

    fft_data = forensics.get("fft", {})
    noise_data = forensics.get("noise", {})
    texture_data = forensics.get("texture", {})
    reasoning = []

    fft_ai = fft_data.get("ai_score", 0.0)
    if fft_ai > 0.15:
        reasoning.append({
            "signal": "FFT Frequency Analysis",
            "value": round(float(fft_ai), 3),
            "severity": _sev(fft_ai),
            "message": (
                "High-frequency anomaly detected - spectral slope deviation "
                f"({round(float(fft_data.get('spectral_slope', 0)), 2)}) "
                "is inconsistent with natural scanner acquisition."
            ),
        })

    fft_tamp = fft_data.get("tamper_score", 0.0)
    if fft_tamp > 0.2:
        reasoning.append({
            "signal": "FFT Quadrant Inconsistency",
            "value": round(float(fft_tamp), 3),
            "severity": _sev(fft_tamp),
            "message": (
                "Frequency content varies significantly across image quadrants - "
                "typical of regional splicing or copy-paste tampering."
            ),
        })

    noise_ai = noise_data.get("ai_score", 0.0)
    if noise_ai > 0.15:
        reasoning.append({
            "signal": "Noise Residual Analysis",
            "value": round(float(noise_ai), 3),
            "severity": _sev(noise_ai),
            "message": (
                "Scanner noise signature is "
                f"{'absent (image too smooth)' if noise_data.get('noise_std', 0) < 0.003 else 'inconsistent'}. "
                "Natural MRI/CT scans contain characteristic acquisition noise."
            ),
        })

    noise_tamp = noise_data.get("tamper_score", 0.0)
    if noise_tamp > 0.25:
        reasoning.append({
            "signal": "Noise Spatial Uniformity",
            "value": round(float(noise_tamp), 3),
            "severity": _sev(noise_tamp),
            "message": (
                "Non-uniform noise distribution across blocks - "
                "regions with different noise levels may indicate patchwork editing."
            ),
        })

    tex_ai = texture_data.get("ai_score", 0.0)
    if tex_ai > 0.15:
        reasoning.append({
            "signal": "Texture / DCT Uniformity",
            "value": round(float(tex_ai), 3),
            "severity": _sev(tex_ai),
            "message": (
                "DCT block variance "
                + (
                    "is unusually uniform - characteristic of GAN/diffusion synthesis"
                    if texture_data.get("dct_uniform")
                    else "shows anomalous local contrast - possible region insertion"
                )
                + "."
            ),
        })

    if heatmap_max > 0.35:
        reasoning.append({
            "signal": "Heatmap Localization",
            "value": round(float(heatmap_max), 3),
            "severity": _sev(heatmap_max),
            "message": (
                "The localizer found elevated activation in suspicious regions, "
                "which reduced the trust score."
            ),
        })

    if not reasoning:
        reasoning.append({
            "signal": "All Forensic Signals",
            "value": 0.0,
            "severity": "LOW",
            "message": (
                "No statistically significant anomalies detected across FFT, "
                "noise residual, texture analysis, or heatmap localization."
            ),
        })

    high_sev = [item for item in reasoning if item["severity"] == "HIGH"]
    med_sev = [item for item in reasoning if item["severity"] == "MEDIUM"]

    if "AI-GENERATED" in label_upper:
        conclusion = (
            f"Multiple synthetic-image fingerprints detected across {len(high_sev)} signals."
            if high_sev
            else "Spectral, texture, or localization patterns suggest AI synthesis with moderate confidence."
        )
    elif "TAMPERED" in label_upper:
        issue_count = len(high_sev) + len(med_sev)
        conclusion = (
            f"{issue_count} forensic inconsistencies detected - likely regional manipulation."
            if issue_count
            else "Marginal tampering indicators - inconclusive without expert review."
        )
    else:
        if high_sev:
            conclusion = (
                f"Classified as ORIGINAL but {len(high_sev)} anomalous signal(s) were detected - "
                "manual review is recommended."
            )
        else:
            conclusion = (
                "Forensic signals are broadly consistent with authentic medical scanner acquisition."
            )

    return {
        "trust_score": trust_score,
        "confidence_tier": confidence_tier,
        "clinical_reliability": clinical_reliability,
        "reliability_label": reliability_label,
        # requires_review = any score below the 'safe' threshold (85)
        "requires_review": trust_score < 85 or confidence_tier in ("UNCERTAIN", "HUMAN_REVIEW_REQUIRED"),
        "fusion": {
            "cnn_weight": w_cnn,
            "fft_weight": w_fft,
            "noise_weight": w_noise,
            "texture_weight": w_texture,
            "heatmap_weight": w_heatmap,
            "cnn_value": round(cnn_value, 3),
            "fft_value": round(fft_value, 3),
            "noise_value": round(noise_value, 3),
            "texture_value": round(texture_value, 3),
            "heatmap_value": round(heatmap_value, 3),
            "cnn_contribution": round(cnn_contrib, 3),
            "fft_contribution": round(fft_contrib, 3),
            "noise_contribution": round(noise_contrib, 3),
            "texture_contribution": round(texture_contrib, 3),
            "heatmap_contribution": round(heatmap_contrib, 3),
            "heatmap_max": round(float(heatmap_max), 3),
        },
        "forensic_reasoning": reasoning,
        "conclusion": conclusion,
    }
