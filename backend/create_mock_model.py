"""
Create a mock cough-classification model for development & testing.

Generates synthetic MFCC-like features with class-separable distributions,
trains a ``RandomForestClassifier``, and serialises it to
``model/cough_model.joblib``.

Usage
-----
    python create_mock_model.py

The resulting model is NOT intended for production — use ``train_model.py``
with real audio data for that.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES_PER_CLASS: int = 100
N_FEATURES: int = 13  # number of MFCC coefficients
N_ESTIMATORS: int = 100
RANDOM_STATE: int = 42
MODEL_DIR: Path = Path(__file__).resolve().parent / "model"
MODEL_PATH: Path = MODEL_DIR / "cough_model.joblib"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_data(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic MFCC-like feature vectors for three classes.

    Each class is drawn from a distinct multivariate normal distribution so
    that the classifier can learn meaningful decision boundaries even on
    fake data.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random number generator for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (3 * N_SAMPLES_PER_CLASS, N_FEATURES)
        Feature matrix.
    y : np.ndarray, shape (3 * N_SAMPLES_PER_CLASS,)
        Label vector with values ``"cough"``, ``"non_cough"``, ``"noise"``.
    """
    # Cough: higher mean values (5–10 range)
    cough_features = rng.normal(
        loc=7.5, scale=1.5, size=(N_SAMPLES_PER_CLASS, N_FEATURES)
    )

    # Non-cough: medium values (0–5 range)
    non_cough_features = rng.normal(
        loc=2.5, scale=1.5, size=(N_SAMPLES_PER_CLASS, N_FEATURES)
    )

    # Noise: lower / negative values (-5–0 range)
    noise_features = rng.normal(
        loc=-2.5, scale=1.5, size=(N_SAMPLES_PER_CLASS, N_FEATURES)
    )

    X = np.vstack([cough_features, non_cough_features, noise_features])
    y = np.array(
        ["cough"] * N_SAMPLES_PER_CLASS
        + ["non_cough"] * N_SAMPLES_PER_CLASS
        + ["noise"] * N_SAMPLES_PER_CLASS
    )
    return X, y


# ---------------------------------------------------------------------------
# Training & persistence
# ---------------------------------------------------------------------------


def create_mock_model() -> None:
    """Generate data, train a Random Forest, and save the model to disk."""
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _generate_synthetic_data(rng)

    print(f"Generated {len(y)} synthetic samples ({N_FEATURES} features each).")
    print(f"  Classes: {np.unique(y).tolist()}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)

    # Training-set accuracy (sanity check — NOT a test-set metric)
    train_acc = clf.score(X, y)
    print(f"  Training accuracy: {train_acc:.4f}")

    # Ensure the model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Done! You can now start the API with: python main.py")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    create_mock_model()
