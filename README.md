# 🫁 DustCough AI

**AI-Powered Cough Sound Screening for Dust Exposure Risk**

DustCough AI is a web-based MVP that records a short cough sound, classifies it using machine learning, and combines the result with PM2.5 air quality data and reported symptoms to provide a preliminary health risk assessment.

> ⚠️ **Disclaimer**: This system is **not** a medical diagnosis tool. It is intended only for preliminary health risk awareness and screening. For health concerns, please consult a healthcare professional.

---

## 📋 Features

- **Cough Sound Recording** — Record 3–5 seconds of audio directly in the browser (WAV format)
- **AI Classification** — Classifies audio as `cough`, `non_cough`, or `noise` using MFCC features + RandomForest
- **PM2.5 & Symptom Assessment** — Collects air quality data, symptoms, outdoor exposure, and mask usage
- **Risk Score Calculation** — Combines AI result, PM2.5, and symptoms into a Low / Medium / High risk level
- **Self-Care Advice** — Provides basic guidance based on risk level

## 🗂️ Project Structure

```
DustCoughAI/
├── backend/
│   ├── main.py                  # FastAPI server (port 8000)
│   ├── create_mock_model.py     # Generate mock model for demo
│   ├── train_model.py           # Train on real dataset
│   ├── requirements.txt         # Python dependencies
│   ├── model/                   # Trained model files (.joblib)
│   └── data/                    # Training audio data
│       ├── cough/
│       ├── non_cough/
│       └── noise/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── RecordPage.jsx
│   │   │   ├── SymptomFormPage.jsx
│   │   │   └── ResultPage.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** installed
- **Node.js 18+** and npm installed
- A modern web browser (Chrome, Firefox, Edge) with microphone access

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate mock model (required for first run)
python create_mock_model.py

# Start the server
python main.py
```

The backend will run at **http://localhost:8000**.

You can verify it's running by visiting http://localhost:8000/health.

### 2. Frontend Setup

```bash
# Navigate to frontend (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run at **http://localhost:5173**.

### 3. Use the App

1. Open **http://localhost:5173** in your browser
2. Click **"Start Screening"**
3. Record a 3–5 second cough sound
4. View the AI classification result
5. Enter PM2.5 value and select symptoms
6. View your risk assessment

---

## 🧠 How It Works

### AI Model

- Extracts **13 MFCC features** from audio using `librosa`
- Classifies using **RandomForestClassifier** from `scikit-learn`
- Labels: `cough`, `non_cough`, `noise`

### Risk Score

| Factor | Condition | Points |
|--------|-----------|--------|
| PM2.5 | < 37.5 µg/m³ | +0 |
| PM2.5 | 37.5–75 µg/m³ | +1 |
| PM2.5 | > 75 µg/m³ | +2 |
| AI Result | Detects cough | +2 |
| Symptoms | 1–2 selected | +1 |
| Symptoms | 3+ selected | +2 |
| Exposure | Long outdoor or no mask | +1 |

| Total Score | Risk Level |
|-------------|------------|
| 0–2 | 🟢 Low |
| 3–5 | 🟡 Medium |
| 6+ | 🔴 High |

---

## 🏋️ Training with Real Data

To train with real audio data:

1. Place audio files (`.wav`) in the appropriate directories:
   - `backend/data/cough/` — cough sounds
   - `backend/data/non_cough/` — speech, breathing, silence
   - `backend/data/noise/` — background noise, environmental sounds

2. Run the training script:
   ```bash
   cd backend
   python train_model.py
   ```

3. The script will output accuracy metrics and save the model to `backend/model/cough_model.joblib`.

---

## 📊 Dataset Information

DustCough AI is developed based on approaches from public cough sound datasets such as **COUGHVID** and **Coswara**, together with additional test samples collected for this project. The system is intended only for preliminary health risk awareness, not medical diagnosis.

### Recommended Datasets

- [COUGHVID](https://coughvid.epfl.ch/) — Large-scale crowdsourced cough recordings
- [Coswara](https://coswara.iisc.ac.in/) — Respiratory sounds for COVID-19 screening research

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite |
| Backend | Python FastAPI |
| AI/ML | librosa, scikit-learn, joblib |
| Audio | Browser MediaRecorder API (WAV) |

---

## 📝 License

This project is for educational and research purposes only.

---

*Built with ❤️ for healthtech innovation*
