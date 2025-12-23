"""
Hazard Identifier node.

Computes qualitative hazard scores in [0, 1] for:
- property_hazard
- liability_exposure

Approach:
- Optionally ask the LLM for a one-sentence rationale to aid explainability.
"""

from typing import Any, Dict, List
import numpy as np

try:
    # joblib comes with scikit-learn
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


# -------------------------- Utilities --------------------------


def _safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _features_from_state(state: Dict[str, Any]) -> np.ndarray:
    """Extract a minimal numeric feature vector."""
    req = state.get("request") or {}
    profile = state.get("profile") or {}

    # Property features
    prop = req.get("property") or {}
    sqft = float(prop.get("sqft") or 0.0)
    sprinklers = 1.0 if bool(prop.get("sprinklers")) else 0.0
    year_built = float(prop.get("year_built") or 1980)

    # Operational features
    employee_count = float(req.get("employee_count") or 0.0)
    revenue = float(req.get("annual_revenue") or 0.0)
    tags: List[str] = list(profile.get("risk_tags") or [])
    public_traffic = 1.0 if "public_foot_traffic" in tags else 0.0
    cooking = 1.0 if "cooking" in tags else 0.0

    # Very small, interpretable vector
    return np.array(
        [
            sqft / 10000.0,        # normalized size
            sprinklers,            # safety feature
            (year_built - 1900) / 200.0,  # newer buildings lower risk
            employee_count / 100.0,
            (revenue / 1e6) / 10.0,  # scale by millions
            public_traffic,
            cooking,
        ],
        dtype=float,
    ).reshape(1, -1)


def _predict_scores_with_model(x: np.ndarray, model) -> Dict[str, float]:
    """
    Map a single logistic regression probability to two related dimensions.
    """
    try:
        proba = float(model.predict_proba(x)[0][1])
    except Exception:
        # Fallback to decision function if needed
        try:
            score = float(model.decision_function(x)[0])
            proba = 1.0 / (1.0 + np.exp(-score))
        except Exception:
            proba = 0.5

    # Split one signal into two related but distinct scores.
    property_hazard = _clip01(proba * 0.9 + 0.05)      # slightly conservative
    liability_exposure = _clip01(0.6 * proba + 0.2)    # smoother mapping

    return {
        "property_hazard": property_hazard,
        "liability_exposure": liability_exposure,
    }


def _heuristic_scores(state: Dict[str, Any]) -> Dict[str, float]:
    """Very small ruleset as a deterministic fallback."""
    req = state.get("request") or {}
    profile = state.get("profile") or {}
    prop = req.get("property") or {}

    base = 0.35
    sqft = float(prop.get("sqft") or 0.0)
    sprinklers = bool(prop.get("sprinklers"))
    year_built = int(prop.get("year_built") or 1980)
    employee_count = int(req.get("employee_count") or 0)
    tags: List[str] = list(profile.get("risk_tags") or [])

    property_hazard = base
    property_hazard += 0.15 if sqft > 20000 else 0.0
    property_hazard += 0.12 if year_built < 1975 else 0.0
    property_hazard -= 0.10 if sprinklers else 0.0
    property_hazard += 0.12 if "cooking" in tags else 0.0
    property_hazard = _clip01(property_hazard)

    liability_exposure = base
    liability_exposure += 0.15 if employee_count > 50 else 0.0
    liability_exposure += 0.12 if "public_foot_traffic" in tags else 0.0
    liability_exposure += 0.07 if "hazmat" in tags else 0.0
    liability_exposure = _clip01(liability_exposure)

    return {
        "property_hazard": property_hazard,
        "liability_exposure": liability_exposure,
    }


def _llm_rationale(*, llm: Any, profile: Dict[str, Any], scores: Dict[str, float]) -> str:
    """
    Ask the LLM for a one-sentence explanation using the profile context.
    """
    name = profile.get("business_name") or "the business"
    ops = profile.get("operations_summary") or "its operations"
    tags = ", ".join(profile.get("risk_tags") or [])
    ph = f"{scores.get('property_hazard', 0):.2f}"
    le = f"{scores.get('liability_exposure', 0):.2f}"

    system = (
        "You are an insurance underwriting assistant. "
        "Explain hazard scores succinctly in one sentence."
    )
    human = (
        f"Business: {name}\nOps: {ops}\nTags: {tags or 'none'}\n"
        f"Scores -> property_hazard: {ph}, liability_exposure: {le}\n"
        "Give one concise sentence suitable for a report."
    )

    try:
        msg = llm.invoke([("system", system), ("human", human)])
        text = getattr(msg, "content", "") or ""
        # Ensure we keep it short
        return (text.strip().split("\n")[0])[:240] or "Scores based on operations and exposure characteristics."
    except Exception:
        return "Scores based on operations and exposure characteristics."


# -------------------------- Node entrypoint --------------------------


def run(*, state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Calculate hazard scores and add them to the state under `hazard_scores`.
    Also stores a short LLM rationale under `hazard_rationale` (scratch).
    """
    scores = _heuristic_scores(state)
    profile = state.get("profile") or {}
    rationale = _llm_rationale(llm=llm, profile=profile, scores=scores)
    new_state = dict(state)
    new_state["hazard_scores"] = scores
    new_state["hazard_rationale"] = rationale
    return new_state
