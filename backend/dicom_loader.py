"""
dicom_loader.py
Loads and parses DICOM files using pydicom.
Returns pixel array and metadata dictionary.

Handles all transfer syntaxes with a multi-strategy fallback chain:
  1. Native pydicom decode (uncompressed or if decoder installed)
  2. Force-override Transfer Syntax to ExplicitVRLittleEndian and decode raw bytes
  3. PIL/Pillow decode of the raw pixel bytes as a last resort
"""
import io
import hashlib
import logging
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    ExplicitVRBigEndian,
    UID,
)

logger = logging.getLogger(__name__)


def _to_number(val):
    """
    Safely convert a pydicom value (DS, IS, or MultiValue) to a Python float.
    Handles single values, multi-value sequences (takes first), and None.
    This is needed because str(MultiValue) in newer pydicom gives Python list
    repr (e.g. "[1119, 400]") instead of DICOM backslash format.
    """
    if val is None:
        return None
    # MultiValue or any sequence: take the first element
    if hasattr(val, '__iter__') and not isinstance(val, str):
        items = list(val)
        if not items:
            return None
        val = items[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# Transfer Syntaxes that are uncompressed and natively supported by pydicom
_UNCOMPRESSED_SYNTAXES = {
    "1.2.840.10008.1.2",        # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",      # Explicit VR Little Endian
    "1.2.840.10008.1.2.1.99",   # Deflated Explicit VR Little Endian
    "1.2.840.10008.1.2.2",      # Explicit VR Big Endian
}

CRITICAL_TAGS = {
    "(0008,0060)": "Modality",
    "(0020,000D)": "StudyInstanceUID",
    "(0020,000E)": "SeriesInstanceUID",
    "(0008,0020)": "StudyDate",
    "(0008,0030)": "StudyTime",
    "(0010,0010)": "PatientName",
    "(0010,0020)": "PatientID",
    "(0010,0030)": "PatientBirthDate",
    "(0010,0040)": "PatientSex",
    "(0008,103E)": "SeriesDescription",
    "(0028,0004)": "PhotometricInterpretation",
    "(0028,0103)": "PixelRepresentation",
    "(0028,0010)": "Rows",
    "(0028,0011)": "Columns",
    "(0028,0030)": "PixelSpacing",
    "(0028,0100)": "BitsAllocated",
    "(0028,0101)": "BitsStored",
    "(0028,1050)": "WindowCenter",
    "(0028,1051)": "WindowWidth",
    "(0008,0016)": "SOPClassUID",
    "(0008,0018)": "SOPInstanceUID",
    "(0008,0070)": "Manufacturer",
    "(0008,1090)": "ManufacturerModelName",
    "(0018,0050)": "SliceThickness",
    "(0020,0013)": "InstanceNumber",
}


def _get_transfer_syntax(ds) -> str | None:
    """Safely extract the Transfer Syntax UID from a dataset."""
    try:
        if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID"):
            return str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        pass
    return None


def _try_native_pixel_array(ds) -> np.ndarray | None:
    """Attempt standard pydicom pixel_array decode."""
    try:
        arr = ds.pixel_array
        return arr
    except Exception as e:
        logger.debug("Native pixel_array failed: %s", e)
        return None


def _try_uncompressed_override(ds) -> np.ndarray | None:
    """
    Override Transfer Syntax to ExplicitVRLittleEndian and re-decode.
    Works for files with wrong/missing Transfer Syntax but raw uncompressed pixels.
    """
    try:
        import copy
        ds_copy = copy.deepcopy(ds)
        if not hasattr(ds_copy, "file_meta") or ds_copy.file_meta is None:
            ds_copy.file_meta = pydicom.dataset.FileMetaDataset()
        ds_copy.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        arr = ds_copy.pixel_array
        logger.info("Pixel array decoded by overriding Transfer Syntax to ExplicitVRLittleEndian")
        return arr
    except Exception as e:
        logger.debug("ExplicitVRLittleEndian override failed: %s", e)

    try:
        import copy
        ds_copy = copy.deepcopy(ds)
        if not hasattr(ds_copy, "file_meta") or ds_copy.file_meta is None:
            ds_copy.file_meta = pydicom.dataset.FileMetaDataset()
        ds_copy.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        arr = ds_copy.pixel_array
        logger.info("Pixel array decoded by overriding Transfer Syntax to ImplicitVRLittleEndian")
        return arr
    except Exception as e:
        logger.debug("ImplicitVRLittleEndian override failed: %s", e)
        return None


def _try_raw_pixel_bytes(ds) -> np.ndarray | None:
    """
    Build a numpy array directly from PixelData bytes using metadata tags.
    Works for uncompressed files that fail all other methods.
    """
    try:
        pixel_data = ds.PixelData
        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))
        bits_allocated = int(getattr(ds, "BitsAllocated", 16))
        pixel_rep      = int(getattr(ds, "PixelRepresentation", 0))  # 0=unsigned, 1=signed
        samples = int(getattr(ds, "SamplesPerPixel", 1))

        if rows <= 0 or cols <= 0:
            return None

        # Respect PixelRepresentation: 1 = signed (int), 0 = unsigned
        dtype_map_unsigned = {8: np.uint8,  16: np.uint16, 32: np.uint32}
        dtype_map_signed   = {8: np.int8,   16: np.int16,  32: np.int32}
        dtype = (dtype_map_signed if pixel_rep == 1 else dtype_map_unsigned).get(bits_allocated, np.uint16)
        bytes_per_pixel = bits_allocated // 8

        raw = bytes(pixel_data)
        expected = rows * cols * samples * bytes_per_pixel
        if len(raw) < expected:
            # Use what we have; pad with zeros
            raw = raw + b"\x00" * (expected - len(raw))
        raw = raw[:expected]

        arr = np.frombuffer(raw, dtype=dtype).reshape((rows, cols) if samples == 1 else (rows, cols, samples))
        logger.info("Pixel array built from raw PixelData bytes — shape=%s dtype=%s", arr.shape, arr.dtype)
        return arr
    except Exception as e:
        logger.debug("Raw pixel bytes decode failed: %s", e)
        return None


def _try_pillow_decode(ds) -> np.ndarray | None:
    """
    Try decoding PixelData as an image using Pillow.
    Works for JPEG-encoded DICOM when pylibjpeg is unavailable.
    """
    try:
        from PIL import Image
        pixel_data = ds.PixelData

        # Strip DICOM encapsulation if present
        raw = bytes(pixel_data)
        # Find JPEG SOI marker (FFD8) or JPEG2000 (FF4F / 0000 0C)
        jpeg_start = raw.find(b"\xff\xd8")
        jp2_start = raw.find(b"\x00\x00\x00\x0c\x6a\x50\x20\x20")  # JP2 signature
        jp2_start2 = raw.find(b"\xff\x4f\xff\x51")  # J2K compressed

        offset = -1
        if jpeg_start != -1:
            offset = jpeg_start
        elif jp2_start != -1:
            offset = jp2_start
        elif jp2_start2 != -1:
            offset = jp2_start2

        if offset == -1:
            return None

        img = Image.open(io.BytesIO(raw[offset:]))
        arr = np.array(img)
        logger.info("Pixel array decoded via Pillow — shape=%s", arr.shape)
        return arr
    except Exception as e:
        logger.debug("Pillow decode failed: %s", e)
        return None


def _extract_pixel_array(ds, transfer_syntax: str | None) -> tuple[np.ndarray | None, str]:
    """
    Try all decoding strategies in sequence. Returns (array, method_used).
    """
    # Strategy 1: Standard pydicom decode
    arr = _try_native_pixel_array(ds)
    if arr is not None:
        return arr, "native"

    # Strategy 2: Override Transfer Syntax (handles missing/wrong TS)
    arr = _try_uncompressed_override(ds)
    if arr is not None:
        return arr, "ts_override"

    # Strategy 3: Pillow JPEG decode (for JPEG-compressed DICOMs)
    arr = _try_pillow_decode(ds)
    if arr is not None:
        return arr, "pillow"

    # Strategy 4: Raw bytes reconstruction (last resort for uncompressed)
    arr = _try_raw_pixel_bytes(ds)
    if arr is not None:
        return arr, "raw_bytes"

    return None, "failed"


def load_dicom(file_bytes: bytes) -> dict:
    """Load DICOM from bytes with robust multi-strategy pixel decoding."""
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Read the dataset — force=True to handle files without proper DICOM preamble
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    except Exception as e:
        raise ValueError(f"Cannot read DICOM file structure: {e}")

    transfer_syntax = _get_transfer_syntax(ds)
    logger.info("DICOM loaded — TransferSyntax: %s", transfer_syntax or "NOT FOUND")

    # Multi-strategy pixel extraction
    pixel_array, decode_method = _extract_pixel_array(ds, transfer_syntax)

    if pixel_array is not None:
        logger.info("Pixel data extracted via [%s] — shape=%s, dtype=%s",
                    decode_method, pixel_array.shape, pixel_array.dtype)
    else:
        logger.warning(
            "All pixel decode strategies failed. TransferSyntax=%s. "
            "Install pylibjpeg + pylibjpeg-openjpeg for compressed DICOM support.",
            transfer_syntax
        )

    # Extract metadata tags
    tags = {}
    present_count = 0
    for tag_str, attr_name in CRITICAL_TAGS.items():
        try:
            val = getattr(ds, attr_name, None)
            if val is not None:
                tags[tag_str] = str(val)
                present_count += 1
            else:
                tags[tag_str] = None
        except Exception:
            tags[tag_str] = None

    return {
        "ds": ds,
        "pixel_array": pixel_array,
        "decode_method": decode_method,
        "transfer_syntax": transfer_syntax,
        "tags": tags,
        "present_count": present_count,
        "total_count": len(CRITICAL_TAGS),
        "file_hash": file_hash,
        # ── Numeric imaging parameters (float, bypasses str() conversion issues)
        "window_center":  _to_number(getattr(ds, "WindowCenter",      None)),
        "window_width":   _to_number(getattr(ds, "WindowWidth",        None)),
        "rescale_slope":  _to_number(getattr(ds, "RescaleSlope",       None)),
        "rescale_intercept": _to_number(getattr(ds, "RescaleIntercept", None)),
        "photometric_interpretation": str(
            getattr(ds, "PhotometricInterpretation", "") or ""
        ).strip().upper(),
    }
