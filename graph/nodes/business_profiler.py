"""
Business Profiler node.

Responsibilities:
- Normalize the incoming request into a compact `profile` dict.
- Use the LLM to (a) infer NAICS if missing/ambiguous, and (b) produce a brief
  operations summary and risk tags useful to downstream nodes.

Output (merged into state under `profile`):
{
  "business_name": str,
  "naics_code": str | None,
  "operations_summary": str,
  "risk_tags": list[str],      # e.g., ["cooking", "public_foot_traffic"]
  "location_notes": str        # coarse, non-geo-sensitive notes
}
"""

import json
from typing import Any, Dict, List, Optional


def _llm_structured_enrichment(
    *,
    llm: Any,
    business_name: str,
    naics_code: Optional[str],
    operations_desc: str,
) -> Dict[str, Any]:
    """
    Ask the LLM for a compact JSON payload with:
    - predicted_naics (string or null),
    - operations_summary (<= 50 words),
    - risk_tags (<= 6 short tokens),
    - issues (<= 3 short tokens).

    We instruct it to return ONLY JSON; we then parse with a conservative
    fallback if the output isn't perfectly valid JSON.
    """
    system = (
        "You are an insurance underwriting assistant. "
        "Return compact JSON only—no prose."
    )
    human = (
        "Normalize the following business information for commercial insurance. "
        "Return a JSON object with keys: "
        "predicted_naics (string|null), operations_summary (string <= 50 words), "
        "risk_tags (array of <= 6 short snake_case tokens), "
        "issues (array of <= 3 short tokens). "
        f'Existing NAICS (if any): "{naics_code}".\n\n'
        f'Business name: "{business_name}"\n'
        f'Operations description: "{operations_desc}"'
    )

    msg = llm.invoke([("system", system), ("human", human)])
    raw = getattr(msg, "content", "") or ""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object.")
    except Exception:
        # Minimal fallback if model didn't return clean JSON.
        data = {
            "predicted_naics": naics_code,
            "operations_summary": operations_desc[:300],
            "risk_tags": [],
            "issues": [],
        }
    # Ensure required keys exist
    data.setdefault("predicted_naics", naics_code)
    data.setdefault("operations_summary", operations_desc[:300])
    data.setdefault("risk_tags", [])
    data.setdefault("issues", [])
    return data


def run(*, state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Enrich the request and attach a normalized `profile` to the state.
    """
    req = state.get("request") or {}

    business_name: str = req.get("business_name") or "Unknown Business"
    naics_code: Optional[str] = req.get("naics_code")
    operations: Dict[str, Any] = req.get("operations") or {}
    operations_desc: str = operations.get("description") or "No description."

    # LLM enrichment (NAICS guess + short ops summary + tags)
    enriched = _llm_structured_enrichment(
        llm=llm,
        business_name=business_name,
        naics_code=naics_code,
        operations_desc=operations_desc,
    )

    # Coarse location notes (non-PII): we avoid storing exact coordinates here.
    locations: List[Dict[str, Any]] = req.get("locations") or []
    multi_site = len(locations) > 1
    has_sprinklers = bool((req.get("property") or {}).get("sprinklers"))
    loc_note = "multi_site" if multi_site else "single_site"
    if has_sprinklers:
        loc_note += "; sprinklers_present"

    profile: Dict[str, Any] = {
        "business_name": business_name,
        "naics_code": enriched.get("predicted_naics") or naics_code,
        "operations_summary": enriched.get("operations_summary", operations_desc),
        "risk_tags": enriched.get("risk_tags", []),
        "location_notes": loc_note,
        # Keep issues for downstream attention (not part of final API schema)
        "issues": enriched.get("issues", []),
    }

    # Merge into state and return.
    new_state = dict(state)
    new_state["profile"] = profile
    return new_state
