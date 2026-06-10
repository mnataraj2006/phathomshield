# PhantomaShield

AI-powered DICOM Medical Image Integrity Platform

## 🚀 Quick Start

### 1. Install Backend Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### 3. Start Both Servers

From the project root:

```powershell
.\start.ps1
```

Or manually:

**Backend (Terminal 1):**
```powershell
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (Terminal 2):**
```powershell
cd frontend
npm run dev
```

Then open: **http://localhost:5173**

---

## 📌 Module Overview

### Module 1 — DICOM Validation & Detection
- Upload any `.dcm` DICOM file
- CNN (ResNet50) classifies: **Original / Tampered / AI-Generated**
- **Grad-CAM** heatmap overlays highlight suspicious regions
- DICOM metadata validated (tags, UIDs, date formats)
- **Trust Score** (0–100%) combining all signals

### Module 2 — Corrupted File Recovery
- Upload a corrupted `.dcm` file
- Corruption type & severity analysis
- **Autoencoder** (PyTorch) reconstructs missing regions
- **OpenCV Inpainting** fills damaged pixel areas
- DICOM metadata tags restored where missing
- Download the recovered `.dcm` file

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite |
| Styling | Vanilla CSS (dark navy + cyan theme) |
| Backend | FastAPI (Python) |
| AI Detection | PyTorch ResNet50 |
| Tamper Localization | Grad-CAM |
| Image Recovery | Autoencoder + OpenCV Inpainting |
| DICOM Processing | pydicom |
| Image Processing | OpenCV + NumPy |
| Metadata Hashing | Python hashlib |

---

## 📁 Project Structure

```
phantomashield/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LandingPage.jsx
│   │   │   ├── DetectModule.jsx
│   │   │   ├── RecoverModule.jsx
│   │   │   ├── UploadZone.jsx
│   │   │   ├── HeatmapViewer.jsx
│   │   │   ├── TrustScoreGauge.jsx
│   │   │   ├── MetadataTable.jsx
│   │   │   ├── RecoveryViewer.jsx
│   │   │   └── CorruptionReport.jsx
│   │   ├── App.jsx
│   │   └── index.css
│   └── vite.config.js
├── backend/
│   ├── main.py              ← FastAPI app + endpoints
│   ├── dicom_loader.py      ← pydicom file loading
│   ├── preprocessor.py      ← NumPy + OpenCV preprocessing
│   ├── detector.py          ← ResNet50 CNN classification
│   ├── localizer.py         ← Grad-CAM heatmap generation
│   ├── validator.py         ← DICOM metadata validation
│   ├── trust_score.py       ← Weighted trust score engine
│   ├── corruption_detector.py ← Module 2 corruption analysis
│   ├── recovery_engine.py   ← Autoencoder + Inpainting
│   ├── metadata_restorer.py ← Tag restoration
│   └── requirements.txt
├── models/                  ← Place pretrained .pth files here
│   ├── resnet_dicom.pth     (optional — auto-detected)
│   └── autoencoder_dicom.pth (optional — auto-detected)
└── start.ps1               ← One-click startup script
```

---

## 🔗 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/validate` | Module 1: Detection + Validation |
| POST | `/recover` | Module 2: Corruption Recovery |
| GET | `/result/{id}` | Fetch cached result |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

---

## ⚠️ Important Notes

- **Approximate Reconstruction**: Recovered images are AI-assisted reconstructions. Never use for clinical diagnosis without verifying against the original source.
- **Models**: Pretrained DICOM-specific model weights (`*.pth`) can be placed in `models/`. The system works without them using ImageNet features + heuristics.
- **File Size**: Maximum upload size is 50 MB.
- **CORS**: Configured for `*` origins in development. Restrict in production.
