"""
metadata_restorer.py
Restores missing or broken DICOM metadata tags using pydicom.
Infers reasonable default values for known missing tags.
"""
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from pydicom.sequence import Sequence
import datetime
import logging

logger = logging.getLogger(__name__)

# Default values for commonly missing tags
DEFAULTS = {
    "PatientName": "ANONYMOUS^PATIENT",
    "PatientID": "UNKNOWN",
    "PatientBirthDate": "19000101",
    "PatientSex": "O",
    "StudyDate": None,          # will use today
    "StudyTime": None,          # will use now
    "StudyInstanceUID": None,   # will generate
    "SeriesInstanceUID": None,  # will generate
    "SOPInstanceUID": None,     # will generate
    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
    "Modality": "OT",
    "Manufacturer": "UNKNOWN",
    "BitsAllocated": 16,
    "BitsStored": 16,
    "HighBit": 15,
    "PixelRepresentation": 0,
    "SamplesPerPixel": 1,
    "PhotometricInterpretation": "MONOCHROME2",
}


def restore_metadata(ds: FileDataset, tags: dict) -> tuple[FileDataset, dict]:
    """
    Attempt to restore missing DICOM metadata tags.
    Modifies ds in-place and returns (ds, restored_dict).
    restored_dict maps tag_name → restored_value.
    """
    now = datetime.datetime.now()
    restored = {}

    for attr_name, default in DEFAULTS.items():
        try:
            current = getattr(ds, attr_name, None)
            if current is None or str(current).strip() == '':
                if default is None:
                    if attr_name == "StudyDate":
                        val = now.strftime("%Y%m%d")
                    elif attr_name == "StudyTime":
                        val = now.strftime("%H%M%S")
                    else:
                        val = generate_uid()
                else:
                    val = default

                try:
                    setattr(ds, attr_name, val)
                    restored[attr_name] = str(val)
                    logger.info("Restored tag %s = %s", attr_name, val)
                except Exception as e:
                    logger.warning("Could not set %s: %s", attr_name, e)
        except Exception as e:
            logger.warning("Error checking %s: %s", attr_name, e)

    return ds, restored
