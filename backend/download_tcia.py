"""
download_tcia.py
=================
Downloads DICOM files from a TCIA .tcia manifest file
using the public NBIA REST API — NO Java, NO NBIA Data Retriever app needed.

Usage:
    python download_tcia.py --manifest "C:/path/to/COVID-19-NY-SBU-manifest_20210810.tcia" --out dicom_data --limit 300

What the .tcia file is:
  A plain text file listing Series Instance UIDs.
  This script reads those UIDs and downloads each CT series
  as a ZIP of DICOMs via the TCIA public REST API.

COVID-19-NY-SBU is chest CT — perfect for training PhantomaShield.
"""
import os
import io
import time
import zipfile
import argparse
import requests
from tqdm import tqdm

# TCIA public REST API (no authentication needed for public collections)
NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
NBIA_DOWNLOAD = f"{NBIA_BASE}/getImage"
NBIA_SERIES_INFO = f"{NBIA_BASE}/getSeriesSize"

HEADERS = {
    "User-Agent": "Mozilla/5.0 PhantomaShield-Downloader/1.0",
    "Accept": "application/octet-stream, */*",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def parse_tcia_manifest(manifest_path: str) -> list[str]:
    """
    Parse a .tcia manifest file and extract Series Instance UIDs.
    
    .tcia format (plain text):
        downloadServerUrl=https://...
        includeAnnotation=true
        noOfrRetry=4
        ...
        1.3.6.1.4.1.14519.5.2.1.99.1071.93...
        1.3.6.1.4.1.14519.5.2.1.99.1071.12...
        ...
    """
    uids = []
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # UIDs start with 1. and are long numeric dot-separated strings
            if line.startswith("1.") and len(line) > 20 and "=" not in line:
                uids.append(line)

    print(f"Manifest: {manifest_path}")
    print(f"Found {len(uids)} Series Instance UIDs\n")
    return uids


def get_series_size(series_uid: str) -> int:
    """Get estimated size of a series in bytes."""
    try:
        r = SESSION.get(
            NBIA_SERIES_INFO,
            params={"SeriesInstanceUID": series_uid},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            return int(data.get("TotalSizeInBytes", 0))
    except Exception:
        pass
    return 0


def download_series(series_uid: str, out_dir: str) -> int:
    """
    Download one DICOM series (ZIP) from TCIA and extract .dcm files.
    Returns number of DICOM files extracted.
    """
    series_dir = os.path.join(out_dir, series_uid[:40].replace(".", "_"))
    os.makedirs(series_dir, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if len(existing) > 0:
        return len(existing)

    try:
        r = SESSION.get(
            NBIA_DOWNLOAD,
            params={"SeriesInstanceUID": series_uid},
            timeout=600,   # CT series = 30-200MB, allow 10 min
            stream=True,
        )
        if not r.ok:
            return 0

        # Stream all bytes into memory buffer
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=131072):  # 128 KB chunks
            if chunk:
                buf.write(chunk)

        buf.seek(0)
        total_bytes = buf.getbuffer().nbytes
        if total_bytes < 200:
            return 0  # Empty / error response

        # Extract DICOMs from ZIP — TCIA often ships files with no extension
        count = 0
        with zipfile.ZipFile(buf) as z:
            members = [m for m in z.namelist() if not m.endswith("/")]
            for member in members:
                basename = os.path.basename(member)
                if not basename:
                    continue
                data = z.read(member)
                if len(data) < 128:
                    continue  # Skip tiny/empty entries
                # Save everything as .dcm — pydicom reads by content, not extension
                out_name = f"slice_{count:04d}.dcm"
                with open(os.path.join(series_dir, out_name), "wb") as f:
                    f.write(data)
                count += 1

        return count

    except requests.exceptions.Timeout:
        return 0
    except zipfile.BadZipFile:
        return 0
    except Exception as e:
        print(f"  [ERR] {series_uid[:25]}: {e}")
        return 0



def download_from_manifest(manifest_path: str, out_dir: str, limit: int = 300, delay: float = 0.5):
    """Full download pipeline from .tcia manifest."""
    os.makedirs(out_dir, exist_ok=True)

    uids = parse_tcia_manifest(manifest_path)
    if not uids:
        print("ERROR: No Series UIDs found in manifest.")
        print("Make sure you selected the correct .tcia manifest file.")
        return

    uids = uids[:limit]
    print(f"Downloading up to {len(uids)} series to: {out_dir}\n")

    total_files = 0
    failed = 0

    pbar = tqdm(uids, desc="Downloading series", unit="series", ncols=80)
    for i, uid in enumerate(pbar):
        pbar.set_postfix({"dcm": total_files, "fail": failed})
        n = download_series(uid, out_dir)
        if n > 0:
            total_files += n
        else:
            failed += 1

        # Polite delay to avoid hammering the server
        if i < len(uids) - 1:
            time.sleep(delay)

    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Series downloaded: {len(uids) - failed}/{len(uids)}")
    print(f"  Total DICOM files: {total_files}")
    print(f"  Failed:           {failed}")
    print(f"  Output:           {out_dir}")
    print(f"\nNext step:")
    print(f"  python build_dataset.py --source {out_dir} --out datasets")


def count_existing(out_dir: str) -> int:
    """Count already-downloaded DICOMs."""
    if not os.path.exists(out_dir):
        return 0
    count = 0
    for root, _, files in os.walk(out_dir):
        count += sum(1 for f in files if f.endswith(".dcm"))
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download TCIA DICOMs from a .tcia manifest file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download up to 300 series from the COVID-19-NY-SBU manifest
  python download_tcia.py --manifest "COVID-19-NY-SBU-manifest_20210810.tcia" --out dicom_data

  # Download only 50 series (quick test)
  python download_tcia.py --manifest "COVID-19-NY-SBU-manifest_20210810.tcia" --out dicom_data --limit 50

  # Check how many files are already downloaded
  python download_tcia.py --manifest "..." --out dicom_data --count-only
        """
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to the .tcia manifest file"
    )
    parser.add_argument(
        "--out", default="dicom_data",
        help="Output directory for downloaded DICOMs (default: dicom_data/)"
    )
    parser.add_argument(
        "--limit", type=int, default=300,
        help="Max number of series to download (default: 300). Each series = ~10-30 DICOM slices."
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay in seconds between series downloads (default: 0.5). Increase if you get rate-limited."
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Just count already-downloaded files without downloading more"
    )
    args = parser.parse_args()

    if args.count_only:
        n = count_existing(args.out)
        print(f"Already downloaded: {n} DICOM files in {args.out}")
    else:
        # Show manifest preview
        try:
            uids_preview = parse_tcia_manifest(args.manifest)
            total_in_manifest = len(uids_preview)
            already = count_existing(args.out)
            print(f"Manifest has {total_in_manifest} series total.")
            print(f"Already downloaded: {already} DICOM files in '{args.out}'")
            print(f"Will download: {min(args.limit, total_in_manifest)} series\n")
        except FileNotFoundError:
            print(f"ERROR: Manifest not found: {args.manifest}")
            print("Please provide the full path to your .tcia file.")
            exit(1)

        download_from_manifest(args.manifest, args.out, args.limit, args.delay)
