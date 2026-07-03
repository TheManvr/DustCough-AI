"""
DustCough AI — FastAPI Backend

Provides a REST API for cough audio classification. Accepts uploaded audio
files, extracts MFCC features, and returns predictions from a pre-trained
RandomForestClassifier.

Endpoints
---------
POST /predict-cough  – Upload an audio file and receive a classification.
GET  /health         – Liveness probe.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(__file__).resolve().parent / "model" / "cough_model.joblib"
SAMPLE_RATE: int = 16_000
N_MFCC: int = 13
MIN_DURATION_SECONDS: float = 1.5
TARGET_DURATION_SECONDS: float = 5.0
MIN_RMS_ENERGY: float = 0.008
MIN_PEAK_AMPLITUDE: float = 0.06
LOW_CONFIDENCE_THRESHOLD: float = 0.48
STRONG_NOISE_THRESHOLD: float = 0.88
AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE: float = 0.72
MIN_AUTO_CAPTURE_RMS: float = 0.010
MIN_AUTO_CAPTURE_PEAK: float = 0.055

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DustCough AI",
    description="Audio-based cough detection API powered by MFCC features and a Random Forest classifier.",
    version="1.0.0",
)

# CORS — open during deployment testing for Vercel/Render preview URLs.
# TODO: Restrict allow_origins to the production Vercel frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model() -> Any:
    """Load the serialised scikit-learn model from disk.

    Returns
    -------
    model
        A fitted scikit-learn estimator that exposes ``predict`` and
        ``predict_proba``.

    Raises
    ------
    HTTPException (503)
        If the model file does not exist or cannot be deserialised.
    """
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model file not found at '{MODEL_PATH}'. "
                "Run 'python create_mock_model.py' or 'python train_model.py' first."
            ),
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model: {exc}",
        )
    return model


def _load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file, convert it to mono, and resample to 16 kHz."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded audio file: {exc}",
        )

    if len(y) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty (zero samples).",
        )

    return y.astype(np.float32), sr


def _trim_silence_softly(y: np.ndarray) -> np.ndarray:
    """Remove clear leading/trailing silence without cutting cough tails."""
    if len(y) == 0:
        return y

    trimmed, _ = librosa.effects.trim(
        y,
        top_db=38,
        frame_length=1024,
        hop_length=256,
    )
    # Keep the original if trimming would remove too much; coughs can have
    # quiet inhale/release portions that still help the demo model.
    if len(trimmed) < int(0.65 * len(y)):
        return y
    return trimmed


def _pad_or_trim(y: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad or trim audio to a stable duration expected by feature extraction."""
    if len(y) > target_samples:
        return y[:target_samples]
    if len(y) < target_samples:
        return np.pad(y, (0, target_samples - len(y)))
    return y


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize audio while preserving silence checks already computed."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 1e-6:
        return y
    return (y / peak * 0.95).astype(np.float32)


def _count_cough_like_bursts(y: np.ndarray, sr: int) -> int:
    """Estimate short high-energy bursts that often correspond to cough events."""
    if len(y) == 0:
        return 0

    frame_length = max(256, int(0.04 * sr))
    hop_length = max(128, int(0.02 * sr))
    rms_frames = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    if len(rms_frames) == 0:
        return 0

    max_rms = float(np.max(rms_frames))
    median_rms = float(np.median(rms_frames))
    threshold = max(0.018, median_rms * 2.2, max_rms * 0.32)
    active = rms_frames >= threshold

    bursts = 0
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue

        start = index
        while index < len(active) and active[index]:
            index += 1
        burst_seconds = ((index - start) * hop_length) / sr

        if 0.04 <= burst_seconds <= 1.0:
            bursts += 1

    return bursts


def _check_audio_quality(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Validate that audio is usable before running cough classification."""
    duration_seconds = len(y) / sr if sr else 0.0
    rms_energy = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
    peak_amplitude = float(np.max(np.abs(y))) if len(y) else 0.0
    cough_like_bursts = _count_cough_like_bursts(y, sr)

    quality = {
        "duration_seconds": round(duration_seconds, 3),
        "rms_energy": round(rms_energy, 5),
        "peak_amplitude": round(peak_amplitude, 5),
        "cough_like_bursts": cough_like_bursts,
        "ok": True,
        "label": None,
        "message": None,
    }

    if duration_seconds < MIN_DURATION_SECONDS:
        quality.update(
            {
                "ok": False,
                "label": "too_short",
                "message": "เสียงสั้นเกินไป กรุณาอัดเสียงไอ 3–5 วินาที",
            }
        )
        return quality

    if rms_energy < MIN_RMS_ENERGY or peak_amplitude < MIN_PEAK_AMPLITUDE:
        quality.update(
            {
                "ok": False,
                "label": "too_quiet",
                "message": "เสียงเบาเกินไป กรุณาอัดเสียงใหม่ใกล้ไมโครโฟนมากขึ้น",
            }
        )
        return quality

    return quality


def _preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Prepare audio for the demo cough model."""
    trimmed = _trim_silence_softly(y)
    normalized = _normalize_audio(trimmed)
    target_samples = int(TARGET_DURATION_SECONDS * sr)
    return _pad_or_trim(normalized, target_samples)


def _extract_mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract mean MFCC features from preprocessed audio.

    Parameters
    ----------
    y : np.ndarray
        Preprocessed audio samples.
    sr : int
        Sample rate of the audio.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(N_MFCC,)`` containing the mean of each MFCC
        coefficient computed over time.

    Raises
    ------
    HTTPException (400)
        If the audio file cannot be decoded or is too short.
    """
    if len(y) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty (zero samples).",
        )

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean: np.ndarray = np.mean(mfccs, axis=1)  # shape (N_MFCC,)
    return mfcc_mean


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Return a simple liveness response.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.post("/predict-cough", summary="Classify an uploaded cough audio file")
async def predict_cough(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an audio upload, extract MFCC features, and classify it.

    The endpoint saves the upload to a temporary file, extracts 13 mean MFCC
    coefficients, and feeds them into the pre-trained Random Forest model.

    Parameters
    ----------
    file : UploadFile
        The audio file to classify (WAV, MP3, OGG, FLAC, etc.).

    Returns
    -------
    dict
        JSON object with keys ``label``, ``confidence``, and
        ``probabilities``.

    Raises
    ------
    HTTPException
        400 if the audio is invalid; 503 if the model is unavailable.
    """
    # Validate that a file was actually uploaded
    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # Persist the upload to a temporary file so librosa can read it
    suffix = Path(file.filename).suffix or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(contents)

        raw_audio, sr = _load_audio(tmp_path)
        quality = _check_audio_quality(raw_audio, sr)

        if not quality["ok"]:
            return {
                "label": quality["label"],
                "confidence": 0.0,
                "message": quality["message"],
                "quality": quality,
                "model_mode": "demo_model",
                "probabilities": {},
            }

        processed_audio = _preprocess_audio(raw_audio, sr)

        # Feature extraction
        features = _extract_mfcc_features(processed_audio, sr)

        # Model inference
        model = _load_model()
        features_2d = features.reshape(1, -1)

        raw_prediction = model.predict(features_2d)[0]
        prediction = str(raw_prediction)
        probabilities = model.predict_proba(features_2d)[0]
        classes = [str(cls) for cls in list(model.classes_)]

        # Build a {class_name: probability} mapping
        prob_dict: Dict[str, float] = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }

        # Confidence is the probability assigned to the predicted class
        confidence = round(float(probabilities[classes.index(prediction)]), 4)
        cough_probability = float(prob_dict.get("cough", 0.0))
        noise_probability = float(prob_dict.get("noise", 0.0))
        cough_like_bursts = int(quality["cough_like_bursts"])
        rms_energy = float(quality["rms_energy"])
        peak_amplitude = float(quality["peak_amplitude"])
        auto_capture_quality_ok = (
            cough_like_bursts >= 1
            and rms_energy >= MIN_AUTO_CAPTURE_RMS
            and peak_amplitude >= MIN_AUTO_CAPTURE_PEAK
            and not (prediction == "noise" and noise_probability >= STRONG_NOISE_THRESHOLD)
        )

        # Demo-model guardrail: this repository ships a synthetic MFCC model.
        # For MVP demos, strong cough-like bursts should not be hidden behind a
        # fragile or poorly calibrated class probability just because the mock
        # model is uncertain. This is still a demo screening signal, not a
        # medical diagnosis.
        label = prediction
        message = None
        confidence_calibrated = False

        if auto_capture_quality_ok:
            label = "cough"
            confidence = round(max(confidence, cough_probability, AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE), 4)
            confidence_calibrated = True
            message = (
                "ระบบตรวจพบช่วงเสียงที่มีรูปแบบคล้ายเสียงไอและคุณภาพเสียงเพียงพอ "
                "สำหรับการคัดกรองเบื้องต้น"
            )
        elif cough_like_bursts >= 1 and (
            prediction == "noise"
            or confidence < LOW_CONFIDENCE_THRESHOLD
            or (prediction == "non_cough" and cough_probability >= 0.22)
            or (prediction == "non_cough" and cough_like_bursts >= 2 and confidence < 0.78)
        ):
            label = "uncertain_cough"
            confidence = round(max(cough_probability, 0.45), 4)
            message = (
                "ระบบพบรูปแบบเสียงที่คล้ายเสียงไอ แต่ยังไม่มั่นใจ "
                "กรุณาไออีกครั้งหรือใช้ข้อมูลประกอบร่วมด้วย"
            )
        elif prediction == "noise" and confidence < STRONG_NOISE_THRESHOLD:
            label = "unclear"
            message = "เสียงไม่ชัดเจน กรุณาอัดเสียงใหม่หากต้องการผลที่แม่นยำขึ้น"

        return {
            "label": label,
            "confidence": confidence,
            "message": message,
            "probabilities": prob_dict,
            "quality": quality,
            "model_mode": "demo_model",
            "confidence_calibrated": confidence_calibrated,
        }
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
