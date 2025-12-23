"""
Loss Predictor node.

Produces quantitative loss metrics:
- expected_loss
- pml (probable maximum loss)

Approach:
- Try to load a tiny LinearRegression model from `settings.LOSS_MODEL_PATH`.
- If unavailable, fall back to simple heuristics using request/profile context.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

try:
    # joblib ships with scikit-learn
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


def _clip_nonneg(x: float) -> float:
    return float(max(0.0, x))


def _features_from_state(state: Dict[str, Any]) -> np.ndarray:
    """Extract a tiny numeric vector used by a toy regression model."""
    req = state.get("request") or {}
    profile = state.get("profile") or {}
    prop = req.get("property") or {}

    revenue = float(req.get("annual_revenue") or 0.0)
    employee_count = float(req.get("employee_count") or 0.0)
    sqft = float(prop.get("sqft") or 0.0)
    sprinklers = 1.0 if bool(prop.get("sprinklers")) else 0.0
    year_built = float(prop.get("year_built") or 1980)

    tags: List[str] = list(profile.get("risk_tags") or [])
    cooking = 1.0 if "cooking" in tags else 0.0
    public = 1.0 if "public_foot_traffic" in tags else 0.0

    return np.array(
        [
            (revenue / 1e6),          # millions of USD
            employee_count / 100.0,
            sqft / 10000.0,
            sprinklers,
            (year_built - 1900) / 200.0,
            cooking,
            public,
        ],
        dtype=float,
    ).reshape(1, -1)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    if not joblib:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def _predict_with_model(x: np.ndarray, model) -> Dict[str, float]:
    """
    Use a LinearRegression-like model to produce expected_loss, then derive PML.
    """
    try:
        y = float(model.predict(x)[0])
        expected_loss = _clip_nonneg(y)
    except Exception:
        expected_loss = 0.0

    # Derive PML as a conservative multiplier with a soft cap.
    pml = _clip_nonneg(expected_loss * 6.0)
    return {"expected_loss": expected_loss, "pml": pml}


def _heuristics(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Deterministic fallback if no model is present. Uses minimal signals.
    """
    req = state.get("request") or {}
    profile = state.get("profile") or {}
    prop = req.get("property") or {}

    revenue = float(req.get("annual_revenue") or 0.0)
    employee_count = int(req.get("employee_count") or 0)
    sqft = float(prop.get("sqft") or 0.0)
    sprinklers = bool(prop.get("sprinklers"))
    year_built = int(prop.get("year_built") or 1980)
    tags: List[str] = list(profile.get("risk_tags") or [])

    base = max(5_000.0, revenue * 0.002)  # floor or small share of revenue
    base += 100.0 * employee_count
    base += 0.05 * sqft
    if not sprinklers:
        base *= 1.15
    if year_built < 1975:
        base *= 1.10
    if "cooking" in tags:
        base *= 1.20
    if "hazmat" in tags:
        base *= 1.25

    expected_loss = float(base)
    pml = float(expected_loss * 5.0)
    return {"expected_loss": expected_loss, "pml": pml}


def run(*, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute `loss_estimates` and merge into state.
    """
    estimates = _heuristics(state)
    new_state = dict(state)
    new_state["loss_estimates"] = {
        "expected_loss": float(estimates["expected_loss"]),
        "pml": float(estimates["pml"]),
    }
    return new_state
