"""
DustCough AI — FastAPI Backend

Provides a REST API for cough audio classification. Accepts uploaded audio
files, extracts MFCC features, and returns predictions from a pre-trained
RandomForestClassifier.

Endpoints
---------
GET  /                - Service info.
POST /predict-cough  – Upload an audio file and receive a classification.
GET  /health         – Liveness probe.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
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
HIGH_CONFIDENCE_THRESHOLD: float = 0.75
STRONG_NOISE_THRESHOLD: float = 0.88
AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE: float = 0.72
MIN_AUTO_CAPTURE_RMS: float = 0.010
MIN_AUTO_CAPTURE_PEAK: float = 0.055

AUDIO_LABEL_TEXT: Dict[str, str] = {
    "dry_cough_like": "ลักษณะคล้ายไอแห้ง",
    "wet_cough_like": "ลักษณะคล้ายไอมีเสมหะ",
    "frequent_cough_like": "ลักษณะคล้ายไอถี่หรือต่อเนื่อง",
    "normal_cough_like": "ลักษณะคล้ายเสียงไอทั่วไป",
    "non_cough": "ไม่พบเสียงไอชัดเจน",
    "noise": "เสียงรบกวน",
    "unclear": "เสียงไม่ชัดเจน",
}
COUGH_AUDIO_LABELS = {
    "dry_cough_like",
    "wet_cough_like",
    "frequent_cough_like",
    "normal_cough_like",
}
SAFETY_NOTICE = "ผลลัพธ์นี้เป็นเพียงการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรค"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("dustcough")

_MODEL_CACHE: Any | None = None
_MODEL_LOAD_ERROR: str | None = None

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


def _load_model() -> Any | None:
    """Return the cached scikit-learn model when available.

    Returns
    -------
    The deployed MVP must keep /predict-cough responsive even when a model
    artifact is missing, so model load failures are logged and handled by the
    request-level heuristic fallback instead of marking the service unavailable.
    """
    global _MODEL_CACHE, _MODEL_LOAD_ERROR

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if _MODEL_LOAD_ERROR is not None:
        return None

    if not MODEL_PATH.is_file():
        _MODEL_LOAD_ERROR = (
            f"Model file not found at '{MODEL_PATH}'. "
            "Run 'python create_mock_model.py' or 'python train_model.py' first."
        )
        logger.warning(_MODEL_LOAD_ERROR)
        return None

    load_started = time.perf_counter()
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    except Exception as exc:
        _MODEL_LOAD_ERROR = f"Failed to load model: {exc}"
        logger.exception(exc)
        return None

    logger.info(
        "model loaded path=%s elapsed=%.3fs",
        MODEL_PATH,
        time.perf_counter() - load_started,
    )
    return _MODEL_CACHE


@app.on_event("startup")
def warm_model_on_startup() -> None:
    """Warm the demo model once so /predict-cough stays lightweight."""
    model = _load_model()
    if model is None:
        logger.warning("model warmup skipped; heuristic fallback remains available")


def _load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file, convert it to mono, and resample to 16 kHz."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode the uploaded audio file: {exc}") from exc

    if len(y) == 0:
        raise ValueError("The uploaded audio file is empty (zero samples).")

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
    ValueError
        If the audio array is empty.
    """
    if len(y) == 0:
        raise ValueError("The uploaded audio file is empty (zero samples).")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean: np.ndarray = np.mean(mfccs, axis=1)  # shape (N_MFCC,)
    return mfcc_mean


def _audio_label_from_legacy(label: str, confidence: float) -> str:
    """Translate legacy model labels into the safer public audio label set."""
    if label == "noise":
        return "noise"
    if label == "non_cough":
        return "non_cough"
    if label == "cough" and confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "normal_cough_like"
    return "unclear"


def _quality_label_from_quality(quality: Dict[str, Any] | None) -> str:
    """Return a compact audio quality label for the frontend."""
    if not quality:
        return "unclear"
    if quality.get("ok"):
        return "good"
    legacy_label = str(quality.get("label") or "")
    if legacy_label in {"too_quiet", "too_short", "noise"}:
        return legacy_label
    return "unclear"


def _cough_features_from_quality(quality: Dict[str, Any] | None) -> Dict[str, Any]:
    """Create the public cough feature block from quality-only data."""
    quality = quality or {}
    return {
        "burst_count": int(quality.get("cough_like_bursts") or 0),
        "duration_sec": round(float(quality.get("duration_seconds") or 0.0), 2),
        "rms_level": round(float(quality.get("rms_energy") or 0.0), 4),
        "peak_level": round(float(quality.get("peak_amplitude") or 0.0), 4),
    }


def _active_audio_segment(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Estimate the active sound span inside an auto-captured cough clip."""
    if len(y) == 0 or sr <= 0:
        return y, 0.0

    frame_length = max(256, int(0.035 * sr))
    hop_length = max(128, int(0.018 * sr))
    rms_frames = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    if len(rms_frames) == 0:
        return y, len(y) / sr

    max_rms = float(np.max(rms_frames))
    median_rms = float(np.median(rms_frames))
    threshold = max(0.012, median_rms * 2.0, max_rms * 0.24)
    active_indices = np.where(rms_frames >= threshold)[0]

    if len(active_indices) == 0:
        return y, len(y) / sr

    start_sample = max(0, int(active_indices[0] * hop_length - frame_length // 2))
    end_sample = min(len(y), int(active_indices[-1] * hop_length + frame_length))
    if end_sample <= start_sample:
        return y, len(y) / sr

    return y[start_sample:end_sample], (end_sample - start_sample) / sr


def _extract_cough_analysis_features(
    y: np.ndarray,
    sr: int,
    quality: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Extract public cough features and internal tone-shape hints."""
    base_features = _cough_features_from_quality(quality)
    active_audio, active_duration = _active_audio_segment(y, sr)
    analysis_audio = active_audio if len(active_audio) else y

    if active_duration > 0:
        base_features["duration_sec"] = round(active_duration, 2)

    spectral_metrics = {
        "centroid_hz": 0.0,
        "rolloff_hz": 0.0,
        "zero_crossing_rate": 0.0,
    }

    if len(analysis_audio) == 0:
        return base_features, spectral_metrics

    try:
        spectral_metrics["centroid_hz"] = round(
            float(np.mean(librosa.feature.spectral_centroid(y=analysis_audio, sr=sr))),
            2,
        )
        spectral_metrics["rolloff_hz"] = round(
            float(np.mean(librosa.feature.spectral_rolloff(y=analysis_audio, sr=sr, roll_percent=0.85))),
            2,
        )
        spectral_metrics["zero_crossing_rate"] = round(
            float(np.mean(librosa.feature.zero_crossing_rate(y=analysis_audio))),
            5,
        )
    except Exception as exc:
        logger.debug("spectral feature extraction skipped: %s", exc)

    return base_features, spectral_metrics


def _resolve_audio_label(
    *,
    label: str,
    confidence: float,
    cough_features: Dict[str, Any],
    spectral_metrics: Dict[str, float] | None = None,
) -> str:
    """Choose a safe cough type label without implying any disease."""
    spectral_metrics = spectral_metrics or {}

    if label == "noise":
        return "noise"
    if label == "non_cough":
        return "non_cough"
    if label != "cough" or confidence < LOW_CONFIDENCE_THRESHOLD:
        return "unclear"

    burst_count = int(cough_features.get("burst_count") or 0)
    duration_sec = float(cough_features.get("duration_sec") or 0.0)
    rms_level = float(cough_features.get("rms_level") or 0.0)
    peak_level = float(cough_features.get("peak_level") or 0.0)
    peak_to_rms = peak_level / max(rms_level, 1e-6)
    centroid_hz = float(spectral_metrics.get("centroid_hz") or 0.0)
    zero_crossing_rate = float(spectral_metrics.get("zero_crossing_rate") or 0.0)

    if burst_count >= 3:
        return "frequent_cough_like"

    short_sharp_sound = (
        duration_sec <= 1.25
        and peak_level >= 0.16
        and peak_to_rms >= 2.8
    )
    if short_sharp_sound or (
        duration_sec <= 1.15
        and peak_to_rms >= 3.8
        and zero_crossing_rate >= 0.045
    ):
        return "dry_cough_like"

    longer_lower_frequency_sound = (
        duration_sec >= 1.25
        and centroid_hz > 0
        and centroid_hz <= 1700
        and zero_crossing_rate <= 0.075
    )
    if longer_lower_frequency_sound or (
        duration_sec >= 1.65
        and centroid_hz > 0
        and centroid_hz <= 2100
        and peak_to_rms < 4.5
    ):
        return "wet_cough_like"

    return "normal_cough_like"


def _build_possible_association(audio_label: str) -> str:
    """Return cautious association text that avoids disease claims."""
    if audio_label == "dry_cough_like":
        return (
            "ลักษณะเสียงที่ระบบตรวจพบคล้ายไอแห้ง อาจสัมพันธ์กับการระคายคอ"
            "หรือการระคายเคืองทางเดินหายใจ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอกและค่า PM2.5"
        )
    if audio_label == "wet_cough_like":
        return (
            "ลักษณะเสียงที่ระบบตรวจพบคล้ายไอมีเสมหะ อาจสัมพันธ์กับการระคายเคืองทางเดินหายใจ"
            "หรือสารระคายเคืองในอากาศ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอกและค่า PM2.5"
        )
    if audio_label == "frequent_cough_like":
        return (
            "ลักษณะเสียงที่ระบบตรวจพบคล้ายไอถี่หรือต่อเนื่อง อาจสัมพันธ์กับการระคายคอ"
            "หรือการระคายเคืองทางเดินหายใจ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอกและค่า PM2.5"
        )
    if audio_label == "normal_cough_like":
        return (
            "ลักษณะเสียงที่ระบบตรวจพบคล้ายเสียงไอทั่วไป ควรใช้ร่วมกับอาการที่ผู้ใช้กรอก"
            " ค่า PM2.5 และบริบทการสัมผัสฝุ่น"
        )
    if audio_label == "non_cough":
        return (
            "ไม่พบเสียงไอชัดเจนจากไฟล์เสียงนี้ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอก"
            "และค่า PM2.5 เพื่อประเมินความเสี่ยงเบื้องต้น"
        )
    if audio_label == "noise":
        return (
            "เสียงมีลักษณะเป็นเสียงรบกวน จึงยังไม่ควรใช้สรุปลักษณะเสียงไอ"
            " ควรบันทึกใหม่ในที่เงียบขึ้นและใช้ร่วมกับอาการที่ผู้ใช้กรอก"
        )
    return (
        "เสียงยังไม่ชัดเจนพอสำหรับสรุปลักษณะเสียงไอ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอก"
        "และค่า PM2.5 หรือบันทึกเสียงใหม่หากต้องการวิเคราะห์อีกครั้ง"
    )


def _safe_prediction_response(
    *,
    label: str = "unclear",
    confidence: float = 0.0,
    message: str = "ไม่สามารถวิเคราะห์เสียงได้ชัดเจน",
    quality: Dict[str, Any] | None = None,
    probabilities: Dict[str, float] | None = None,
    confidence_calibrated: bool = False,
    model_loaded: bool = False,
    model_mode: str = "demo_model",
    audio_label: str | None = None,
    audio_quality: str | None = None,
    cough_features: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a stable HTTP 200 prediction response for the frontend."""
    public_audio_label = audio_label or _audio_label_from_legacy(label, float(confidence))
    public_cough_features = cough_features or _cough_features_from_quality(quality)
    public_audio_quality = audio_quality or _quality_label_from_quality(quality)

    return {
        "label": label,
        "audio_label": public_audio_label,
        "cough_detected": public_audio_label in COUGH_AUDIO_LABELS,
        "confidence": round(float(confidence), 4),
        "cough_type_text": AUDIO_LABEL_TEXT.get(public_audio_label, AUDIO_LABEL_TEXT["unclear"]),
        "audio_quality": public_audio_quality,
        "cough_features": public_cough_features,
        "possible_association": _build_possible_association(public_audio_label),
        "safety_notice": SAFETY_NOTICE,
        "message": message,
        "probabilities": probabilities or {},
        "quality": quality or {},
        "model_mode": model_mode,
        "model_loaded": model_loaded,
        "confidence_calibrated": confidence_calibrated,
    }


def _normalize_prediction_label(prediction: str) -> str:
    """Map model-specific labels into the small public API label set."""
    if prediction in {"cough", "non_cough", "noise", "too_quiet", "too_short", "unclear"}:
        return prediction
    return "unclear"


def _demo_heuristic_prediction(
    quality: Dict[str, Any],
    cough_features: Dict[str, Any] | None = None,
    spectral_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Fallback cough screening when the demo model artifact is unavailable."""
    cough_like_bursts = int(quality.get("cough_like_bursts") or 0)
    rms_energy = float(quality.get("rms_energy") or 0.0)
    peak_amplitude = float(quality.get("peak_amplitude") or 0.0)
    public_cough_features = cough_features or _cough_features_from_quality(quality)

    if cough_like_bursts >= 1 and rms_energy >= MIN_AUTO_CAPTURE_RMS and peak_amplitude >= MIN_AUTO_CAPTURE_PEAK:
        return _safe_prediction_response(
            label="cough",
            confidence=AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE,
            message=(
                "ระบบตรวจพบช่วงเสียงที่มีรูปแบบคล้ายเสียงไอและคุณภาพเสียงเพียงพอ "
                "สำหรับการคัดกรองเบื้องต้น"
            ),
            quality=quality,
            probabilities={"cough": AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE},
            confidence_calibrated=True,
            model_loaded=False,
            model_mode="demo_heuristic_fallback",
            audio_label=_resolve_audio_label(
                label="cough",
                confidence=AUTO_CAPTURE_ACCEPTANCE_CONFIDENCE,
                cough_features=public_cough_features,
                spectral_metrics=spectral_metrics,
            ),
            cough_features=public_cough_features,
        )

    if cough_like_bursts >= 1:
        return _safe_prediction_response(
            label="unclear",
            confidence=0.45,
            message="พบรูปแบบเสียงที่คล้ายเสียงไอ แต่ยังไม่ชัดเจน กรุณาลองใหม่อีกครั้ง",
            quality=quality,
            probabilities={"unclear": 0.45},
            model_loaded=False,
            model_mode="demo_heuristic_fallback",
            audio_label="unclear",
            cough_features=public_cough_features,
        )

    return _safe_prediction_response(
        label="non_cough",
        confidence=0.55,
        message="ยังไม่พบรูปแบบเสียงไอที่ชัดเจน",
        quality=quality,
        probabilities={"non_cough": 0.55},
        model_loaded=False,
        model_mode="demo_heuristic_fallback",
        audio_label="non_cough",
        cough_features=public_cough_features,
    )


def _log_final_prediction(response: Dict[str, Any], request_started: float) -> None:
    """Log the final prediction in a consistent shape for Render logs."""
    logger.info(
        "predict-cough final label=%s final_confidence=%.4f total_processing_time=%.3fs",
        response.get("label"),
        float(response.get("confidence") or 0.0),
        time.perf_counter() - request_started,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Backend service info")
def root() -> Dict[str, str]:
    """Return basic API service links for deployment smoke checks."""
    return {
        "name": "DustCough AI Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


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
    """Accept an audio upload and return a safe cough-screening result.

    Audio/model problems intentionally return HTTP 200 with a conservative
    label so the frontend can guide the user instead of treating the backend
    as unavailable.
    """
    request_started = time.perf_counter()
    logger.info(
        "predict-cough request received filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )

    if file.filename is None or file.filename.strip() == "":
        response = _safe_prediction_response(
            label="unclear",
            confidence=0.0,
            message="ไม่สามารถวิเคราะห์เสียงได้ชัดเจน",
            model_mode="input_validation",
        )
        _log_final_prediction(response, request_started)
        return response

    tmp_path: str | None = None
    try:
        contents = await file.read()
        logger.info("predict-cough audio bytes size=%s", len(contents))

        if len(contents) == 0:
            response = _safe_prediction_response(
                label="too_short",
                confidence=0.0,
                message="เสียงสั้นเกินไป กรุณาอัดเสียงไอ 3–5 วินาที",
                model_mode="quality_check",
            )
            _log_final_prediction(response, request_started)
            return response

        # Persist only the captured cough segment so librosa can decode it.
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        raw_audio, sr = _load_audio(tmp_path)
        logger.info(
            "predict-cough audio loaded successfully duration=%.3fs samples=%s elapsed=%.3fs",
            len(raw_audio) / sr if sr else 0.0,
            len(raw_audio),
            time.perf_counter() - request_started,
        )

        quality = _check_audio_quality(raw_audio, sr)
        cough_features, spectral_metrics = _extract_cough_analysis_features(raw_audio, sr, quality)
        cough_like_burst = bool(int(quality["cough_like_bursts"]))
        logger.info(
            "predict-cough audio quality duration=%.3fs rms=%.5f peak=%.5f cough_like_burst=%s",
            float(quality["duration_seconds"]),
            float(quality["rms_energy"]),
            float(quality["peak_amplitude"]),
            cough_like_burst,
        )

        if not quality["ok"]:
            response = _safe_prediction_response(
                label=quality["label"],
                confidence=0.0,
                message=quality["message"],
                quality=quality,
                model_mode="quality_check",
                audio_label="unclear",
                cough_features=cough_features,
            )
            logger.info("predict-cough model loaded=false")
            _log_final_prediction(response, request_started)
            return response

        model = _load_model()
        model_loaded = model is not None
        logger.info("predict-cough model loaded=%s", str(model_loaded).lower())

        if model is None:
            logger.warning("Model unavailable, using demo heuristic fallback")
            response = _demo_heuristic_prediction(quality, cough_features, spectral_metrics)
            _log_final_prediction(response, request_started)
            return response

        processed_audio = _preprocess_audio(raw_audio, sr)

        features = _extract_mfcc_features(processed_audio, sr)
        logger.info(
            "predict-cough features extracted elapsed=%.3fs",
            time.perf_counter() - request_started,
        )

        try:
            features_2d = features.reshape(1, -1)
            raw_prediction = model.predict(features_2d)[0]
            prediction = str(raw_prediction)
            probabilities = model.predict_proba(features_2d)[0]
            classes = [str(cls) for cls in list(model.classes_)]

            prob_dict: Dict[str, float] = {
                cls: round(float(prob), 4)
                for cls, prob in zip(classes, probabilities)
            }

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

            label = _normalize_prediction_label(prediction)
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
            elif label == "cough" and confidence < LOW_CONFIDENCE_THRESHOLD:
                label = "unclear"
                message = "เสียงยังไม่ชัดเจนพอสำหรับสรุปลักษณะเสียงไอ กรุณาลองใหม่อีกครั้ง"
            elif cough_like_bursts >= 1 and (
                label == "unclear"
                or confidence < LOW_CONFIDENCE_THRESHOLD
                or (label == "non_cough" and cough_probability >= 0.22)
                or (label == "non_cough" and cough_like_bursts >= 2 and confidence < 0.78)
            ):
                label = "unclear"
                confidence = round(max(cough_probability, 0.45), 4)
                message = "พบรูปแบบเสียงที่คล้ายเสียงไอ แต่ยังไม่ชัดเจน กรุณาลองใหม่อีกครั้ง"
            elif label == "unclear":
                message = "ไม่สามารถวิเคราะห์เสียงได้ชัดเจน"
            elif label == "non_cough":
                message = "ยังไม่พบรูปแบบเสียงไอที่ชัดเจน"

            if label == "noise" and message is None:
                message = "เสียงมีลักษณะเป็นเสียงรบกวน กรุณาลองบันทึกใหม่ในที่เงียบขึ้น"

            audio_label = _resolve_audio_label(
                label=label,
                confidence=confidence,
                cough_features=cough_features,
                spectral_metrics=spectral_metrics,
            )

            response = _safe_prediction_response(
                label=label,
                confidence=confidence,
                message=message or "วิเคราะห์เสียงสำเร็จ",
                quality=quality,
                probabilities=prob_dict,
                confidence_calibrated=confidence_calibrated,
                model_loaded=True,
                model_mode="demo_model",
                audio_label=audio_label,
                cough_features=cough_features,
            )
        except Exception as exc:
            logger.exception(exc)
            logger.warning("Model unavailable, using demo heuristic fallback")
            response = _demo_heuristic_prediction(quality, cough_features, spectral_metrics)

        _log_final_prediction(response, request_started)
        return response
    except ValueError as exc:
        logger.exception(exc)
        response = _safe_prediction_response(
            label="unclear",
            confidence=0.0,
            message="ไม่สามารถวิเคราะห์เสียงได้ชัดเจน",
            model_mode="audio_decode_error",
        )
        _log_final_prediction(response, request_started)
        return response
    except Exception as exc:
        logger.exception(exc)
        response = _safe_prediction_response(
            label="unclear",
            confidence=0.0,
            message="เกิดข้อผิดพลาดในการวิเคราะห์เสียง กรุณาลองใหม่อีกครั้ง",
            confidence_calibrated=False,
            model_mode="unexpected_error",
        )
        _log_final_prediction(response, request_started)
        return response
    finally:
        # Clean up the temporary file
        if tmp_path is not None and os.path.exists(tmp_path):
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
