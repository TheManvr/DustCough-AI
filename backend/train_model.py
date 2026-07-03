"""
Train a cough-classification model on real audio data.

Reads WAV/MP3/OGG/FLAC files from three subdirectories under ``data/``:

- ``data/cough/``      – cough audio samples
- ``data/non_cough/``  – other human sounds (speech, breathing, etc.)
- ``data/noise/``      – environmental / background noise

Each file is converted to mono at 16 kHz, 13 mean MFCC coefficients are
extracted, and a ``RandomForestClassifier`` is trained on an 80/20 split.
The fitted model is serialised to ``model/cough_model.joblib``.

Usage
-----
    python train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "model"
MODEL_PATH: Path = MODEL_DIR / "cough_model.joblib"

SAMPLE_RATE: int = 16_000
N_MFCC: int = 13
TEST_SIZE: float = 0.20
N_ESTIMATORS: int = 200
RANDOM_STATE: int = 42

# Audio file extensions to look for
AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

# Mapping from directory name → class label
CLASS_DIRS: dict[str, str] = {
    "cough": "cough",
    "non_cough": "non_cough",
    "noise": "noise",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_mfcc(file_path: Path) -> np.ndarray | None:
    """Load an audio file and return the mean of 13 MFCC coefficients.

    Parameters
    ----------
    file_path : Path
        Path to an audio file.

    Returns
    -------
    np.ndarray or None
        1-D array of shape ``(N_MFCC,)`` on success, or ``None`` if the
        file could not be decoded.
    """
    try:
        y, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        print(f"  [WARN] Skipping {file_path.name}: {exc}")
        return None

    if len(y) == 0:
        print(f"  [WARN] Skipping {file_path.name}: zero-length audio")
        return None

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfccs, axis=1)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Walk the data directories and build feature / label arrays.

    Returns
    -------
    X : np.ndarray, shape (n_samples, N_MFCC)
    y : np.ndarray, shape (n_samples,)

    Raises
    ------
    SystemExit
        If required directories are missing or no audio files are found.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []

    for dir_name, label in CLASS_DIRS.items():
        class_dir = DATA_DIR / dir_name

        if not class_dir.is_dir():
            print(
                f"[ERROR] Data directory not found: {class_dir}\n"
                f"        Create it and add audio files before training."
            )
            sys.exit(1)

        audio_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ]

        if len(audio_files) == 0:
            print(
                f"[ERROR] No audio files found in: {class_dir}\n"
                f"        Supported extensions: {sorted(AUDIO_EXTENSIONS)}"
            )
            sys.exit(1)

        print(f"Loading '{label}' — {len(audio_files)} files from {class_dir}")

        for audio_file in sorted(audio_files):
            mfcc_vec = extract_mfcc(audio_file)
            if mfcc_vec is not None:
                features.append(mfcc_vec)
                labels.append(label)

    if len(features) == 0:
        print("[ERROR] No valid audio samples could be loaded. Aborting.")
        sys.exit(1)

    X = np.array(features)
    y = np.array(labels)
    print(f"\nTotal samples loaded: {len(y)}")
    for label in sorted(set(labels)):
        count = sum(1 for l in labels if l == label)
        print(f"  {label}: {count}")

    return X, y


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train() -> None:
    """End-to-end training: load data, train model, evaluate, and save."""
    print("=" * 60)
    print("DustCough AI — Model Training")
    print("=" * 60)
    print()

    # 1. Load dataset
    X, y = load_dataset()

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")

    # 3. Train classifier
    print(f"\nTraining RandomForestClassifier (n_estimators={N_ESTIMATORS}) ...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    # Pretty-print with class labels
    header = "           " + "  ".join(f"{c:>10}" for c in clf.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>10}" for val in row)
        print(f"{clf.classes_[i]:>10}  {row_str}")
    print()

    # 5. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
