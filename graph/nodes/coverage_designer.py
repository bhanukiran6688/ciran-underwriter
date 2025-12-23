"""
Coverage Designer node.

Combines hazard scores and loss estimates to produce a simple recommendation:
- coverages (list[str])
- policy_limits (dict[str, float])
- deductibles (dict[str, float])
- pricing_inputs (dict[str, float])
- rationale (str)  # short explanation from the LLM

The rule set is intentionally small to keep the POC readable.
"""

from typing import Any, Dict, List


def _design_coverages(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very small heuristic "program" to map scores to limits/deductibles.
    """
    hazard = state.get("hazard_scores") or {}
    loss = state.get("loss_estimates") or {}
    req = state.get("request") or {}

    ph = float(hazard.get("property_hazard") or 0.4)
    le = float(hazard.get("liability_exposure") or 0.4)

    expected_loss = float(loss.get("expected_loss") or 10_000.0)
    pml = float(loss.get("pml") or expected_loss * 5.0)

    employee_count = int(req.get("employee_count") or 0)
    revenue = float(req.get("annual_revenue") or 0.0)

    coverages: List[str] = []
    policy_limits: Dict[str, float] = {}
    deductibles: Dict[str, float] = {}

    # Property
    if ph > 0.55 or pml > 100_000:
        coverages.append("Property")
        policy_limits["Property"] = max(250_000.0, pml * 1.25)
        deductibles["Property"] = max(1_000.0, expected_loss * 0.05)

    # General Liability
    if le > 0.55 or employee_count > 50:
        coverages.append("General Liability")
        policy_limits["General Liability"] = 1_000_000.0 if le < 0.75 else 2_000_000.0
        deductibles["General Liability"] = 1_000.0 if le < 0.75 else 2_500.0

    # Cyber (simple trigger based on revenue and exposure)
    if revenue > 1_000_000.0 or le > 0.65:
        coverages.append("Cyber")
        policy_limits["Cyber"] = 250_000.0 if revenue < 5_000_000.0 else 1_000_000.0
        deductibles["Cyber"] = 2_500.0

    if not coverages:
        # Always propose at least GL with modest limits so the POC returns something useful.
        coverages = ["General Liability"]
        policy_limits["General Liability"] = 1_000_000.0
        deductibles["General Liability"] = 1_000.0

    pricing_inputs = {
        "hazard_factor": 0.5 * ph + 0.5 * le,
        "loss_load": expected_loss,
        "pml_load": pml,
    }

    return {
        "coverages": coverages,
        "policy_limits": policy_limits,
        "deductibles": deductibles,
        "pricing_inputs": pricing_inputs,
    }


def _llm_rationale(*, llm: Any, state: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """
    Ask the LLM to write a brief, plain-English rationale.
    """
    profile = state.get("profile") or {}
    name = profile.get("business_name") or "the business"
    ops = profile.get("operations_summary") or ""
    ph = float((state.get("hazard_scores") or {}).get("property_hazard") or 0.0)
    le = float((state.get("hazard_scores") or {}).get("liability_exposure") or 0.0)
    el = float((state.get("loss_estimates") or {}).get("expected_loss") or 0.0)
    pml = float((state.get("loss_estimates") or {}).get("pml") or 0.0)
    covs = ", ".join(rec.get("coverages") or [])

    system = (
        "You are an insurance underwriting assistant. "
        "Write one or two short sentences suitable for a customer-facing rationale."
    )
    human = (
        f"Business: {name}\n"
        f"Ops: {ops}\n"
        f"Hazard scores: property={ph:.2f}, liability={le:.2f}\n"
        f"Loss: expected={el:,.0f}, pml={pml:,.0f}\n"
        f"Recommended coverages: {covs}\n"
        "Keep it concise and factual."
    )

    try:
        msg = llm.invoke([("system", system), ("human", human)])
        text = getattr(msg, "content", "") or ""
        return (text.strip().split("\n")[0])[:400] or "Recommendation based on exposure and loss potential."
    except Exception:
        return "Recommendation based on exposure and loss potential."


def run(*, state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Compose the final recommendation and merge into state.
    """
    rec = _design_coverages(state)
    rationale = _llm_rationale(llm=llm, state=state, rec=rec)
    rec["rationale"] = rationale

    new_state = dict(state)
    new_state["recommendation"] = rec
    return new_state
