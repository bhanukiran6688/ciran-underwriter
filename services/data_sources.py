"""
Simulated external data sources.

For the POC we avoid real HTTP calls and provide deterministic stubs that
downstream nodes can import. If you later connect real services, replace the
stub functions while keeping the signatures.
"""

from typing import Any, Dict, List


def fetch_industry_hazards(*, naics_code: str | None) -> List[str]:
    """
    Return coarse hazard tags by NAICS. Deterministic and tiny on purpose.
    """
    if not naics_code:
        return []
    prefix = naics_code[:2]

    mapping = {
        "72": ["cooking", "public_foot_traffic"],         # Accommodation & Food
        "44": ["public_foot_traffic"],                    # Retail trade
        "23": ["contractor_tools", "work_at_height"],     # Construction
        "31": ["flammables", "machinery"],                # Manufacturing
        "42": ["warehouse_racking", "forklifts"],         # Wholesale
    }
    return mapping.get(prefix, [])


def fetch_location_signals(*, profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Very small placeholder for location-based adjustments.
    Returns multipliers rather than absolutes.
    """
    loc_notes = (profile or {}).get("location_notes", "")
    multi = 1.0
    if "multi_site" in loc_notes:
        multi *= 1.05
    if "sprinklers_present" in loc_notes:
        multi *= 0.95
    return {"location_multiplier": multi}


def fetch_loss_benchmarks(*, naics_code: str | None) -> Dict[str, float]:
    """
    Tiny, hard-coded industry benchmarks to seed loss estimates.
    """
    if not naics_code:
        return {"el_per_million_revenue": 2_000.0, "pml_multiplier": 5.0}

    prefix = naics_code[:2]
    table = {
        "72": {"el_per_million_revenue": 4_000.0, "pml_multiplier": 6.0},
        "44": {"el_per_million_revenue": 2_500.0, "pml_multiplier": 5.5},
        "23": {"el_per_million_revenue": 3_500.0, "pml_multiplier": 6.5},
        "31": {"el_per_million_revenue": 3_000.0, "pml_multiplier": 6.0},
        "42": {"el_per_million_revenue": 2_200.0, "pml_multiplier": 5.0},
    }
    return table.get(prefix, {"el_per_million_revenue": 2_000.0, "pml_multiplier": 5.0})
