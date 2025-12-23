"""
Data schemas for request/response validation.

These models are used by the FastAPI layer and also as a contract for the
workflow's final output.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Address(BaseModel):
    """Basic address information; coordinates are optional."""

    model_config = ConfigDict(extra="forbid")

    line1: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: str = Field(default="US", min_length=2, max_length=2)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lng: Optional[float] = Field(default=None, ge=-180, le=180)


class PropertyInfo(BaseModel):
    """Lightweight property characteristics used by risk nodes."""

    model_config = ConfigDict(extra="forbid")

    construction: Optional[str] = None
    year_built: Optional[int] = Field(default=None, ge=1800, le=2100)
    sqft: Optional[int] = Field(default=None, ge=0)
    sprinklers: Optional[bool] = None


class OperationsInfo(BaseModel):
    """Business operations metadata."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(..., min_length=3)
    hours_per_week: Optional[int] = Field(default=None, ge=0, le=168)


class UnderwritingRequest(BaseModel):
    """
    Incoming underwriting request. Keep intentionally small for the POC while
    still being realistic enough for the nodes to operate on.
    """

    model_config = ConfigDict(extra="forbid")

    business_name: str = Field(..., min_length=1)
    naics_code: Optional[str] = None
    annual_revenue: Optional[float] = Field(default=None, ge=0)
    employee_count: Optional[int] = Field(default=None, ge=0)
    locations: List[Address] = Field(default_factory=list)
    property: Optional[PropertyInfo] = None
    operations: OperationsInfo

    @field_validator("naics_code")
    @classmethod
    def _strip_naics(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() or None if isinstance(v, str) else v


class HazardScores(BaseModel):
    """Qualitative/relative hazard scoring produced by the hazard node."""

    model_config = ConfigDict(extra="forbid")

    property_hazard: float = Field(..., ge=0.0, le=1.0)
    liability_exposure: float = Field(..., ge=0.0, le=1.0)


class LossEstimates(BaseModel):
    """Quantitative loss metrics produced by the loss node."""

    model_config = ConfigDict(extra="forbid")

    expected_loss: float = Field(..., ge=0.0)
    pml: float = Field(..., ge=0.0, description="Probable maximum loss")


class Recommendation(BaseModel):
    """
    The final underwriting recommendation including coverages, limits,
    deductibles, and pricing inputs. A rationale string is included and can be
    produced with an LLM for readability.
    """

    model_config = ConfigDict(extra="forbid")

    coverages: List[str]
    policy_limits: Dict[str, float]
    deductibles: Dict[str, float]
    pricing_inputs: Dict[str, float]
    rationale: str


class UnderwritingResponse(BaseModel):
    """
    The API response model. The workflow should output exactly this shape,
    which we validate before returning to the client.
    """

    model_config = ConfigDict(extra="forbid")

    hazard_scores: HazardScores
    loss_estimates: LossEstimates
    recommendation: Recommendation
