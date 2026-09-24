"""Pydantic schemas for FastAPI request and response validation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HouseFeaturesInput(BaseModel):
    """Schema for individual property features input with strict validation."""
    
    OverallQual: int = Field(
        ..., ge=1, le=10,
        description="Overall material and finish of the house (1=Very Poor, 10=Very Excellent)"
    )
    GrLivArea: float = Field(
        ..., gt=0, le=10000.0,
        description="Above ground living area in square feet"
    )
    TotalBsmtSF: float = Field(
        ..., ge=0, le=6000.0,
        description="Total square feet of basement area"
    )
    GarageCars: int = Field(
        ..., ge=0, le=5,
        description="Size of garage in car capacity"
    )
    FullBath: int = Field(
        ..., ge=1, le=6,
        description="Full bathrooms above ground"
    )
    YearBuilt: int = Field(
        ..., ge=1850, le=2030,
        description="Original construction date"
    )
    YearRemodAdd: int = Field(
        ..., ge=1950, le=2030,
        description="Remodel date"
    )
    LotArea: float = Field(
        ..., gt=0, le=100000.0,
        description="Lot size in square feet"
    )
    Fireplaces: int = Field(
        ..., ge=0, le=5,
        description="Number of fireplaces"
    )
    Neighborhood: Literal[
        "NorthAmes", "CollegeCreek", "OldTown", "Edwards", "Somerset",
        "Gilbert", "Sawyer", "Northridge", "Crawford", "Mitchell"
    ] = Field(
        ...,
        description="Physical location within the city"
    )
    BldgType: Literal["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"] = Field(
        ...,
        description="Type of dwelling (Single family, Townhouse, etc.)"
    )
    HouseStyle: Literal["1Story", "2Story", "1.5Fin", "SLvl"] = Field(
        ...,
        description="Style of dwelling (1 Story, 2 Story, Split Level, etc.)"
    )
    CentralAir: Literal["Y", "N"] = Field(
        ...,
        description="Central air conditioning (Y=Yes, N=No)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1850.0,
                "TotalBsmtSF": 1200.0,
                "GarageCars": 2,
                "FullBath": 2,
                "YearBuilt": 2008,
                "YearRemodAdd": 2015,
                "LotArea": 9500.0,
                "Fireplaces": 1,
                "Neighborhood": "CollegeCreek",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "CentralAir": "Y"
            }
        }
    }


class BatchHouseFeaturesInput(BaseModel):
    """Schema for batch prediction requests."""
    properties: List[HouseFeaturesInput] = Field(..., description="List of property features for inference")


class PredictionOutput(BaseModel):
    """Schema for prediction response."""
    predicted_price: float = Field(..., description="Estimated sale price in USD")
    formatted_price: str = Field(..., description="Formatted sale price with currency symbol")
    currency: str = Field("USD", description="Currency of prediction")
    model_name: str = Field(..., description="Name of the model used for inference")
    model_version: str = Field("1.0.0", description="Version of the model")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionOutput]
    total_count: int


class HealthResponse(BaseModel):
    """Schema for service health check response."""
    status: str
    service: str
    version: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema for model metadata response."""
    model_name: str
    model_version: str
    run_id: str
    training_date: str
    features: Dict[str, List[str]]
    validation_metrics: Dict[str, Any]
    all_model_comparison: Optional[Dict[str, Any]] = None
