"""
API models for request and response payloads.
"""

from datetime import date
from pydantic import BaseModel, Field
from typing import Literal


class HouseFeatures(BaseModel):
    bedrooms: int = Field(
        ...,
        description="Number of bedrooms",
        examples=[0, 1, 2, 3],
        ge=0,
        frozen=True,
        strict=True,
    )
    bathrooms: float = Field(
        ...,
        description="Number of bathrooms",
        examples=[0, 1, 1.5, 2, 2.25],
        ge=0.0,
        multiple_of=0.25,
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )
    sqft_living: int = Field(
        ...,
        description="Square footage of the living space",
        examples=[2000],
        gt=0,
        frozen=True,
        strict=True,
    )
    sqft_lot: int = Field(
        ...,
        description="Square footage of the lot",
        examples=[5000, 10000],
        gt=0,
        frozen=True,
        strict=True,
    )
    floors: float = Field(
        ...,
        description="Number of floors",
        examples=[1.0, 1.5, 2.0],
        ge=1.0,
        multiple_of=0.5,
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )
    waterfront: int = Field(
        ...,
        description="Whether the property has a waterfront view (0 = no, 1 = yes)",
        examples=[0, 1],
        ge=0,
        le=1,
        frozen=True,
        strict=True,
    )
    view: int = Field(
        ...,
        description="View rating (0-4)",
        examples=[0, 1, 2, 3, 4],
        ge=0,
        le=4,
        frozen=True,
        strict=True,
    )
    condition: int = Field(
        ...,
        description="Condition rating (1-5)",
        examples=[1, 2, 3, 4, 5],
        ge=1,
        le=5,
        frozen=True,
        strict=True,
    )
    grade: int = Field(
        ...,
        description="Grade rating (1-13)",
        examples=[1, 5, 10, 13],
        ge=1,
        le=13,
        frozen=True,
        strict=True,
    )
    sqft_above: int = Field(
        ...,
        description="Square footage of the house apart from basement",
        examples=[1500],
        gt=0,
        frozen=True,
        strict=True,
    )
    sqft_basement: int = Field(
        ...,
        description="Square footage of the basement",
        examples=[0, 500],
        ge=0,
        frozen=True,
        strict=True,
    )
    yr_built: int = Field(
        ...,
        description="Year the house was built",
        examples=[1990, 2005],
        ge=1800,
        le=date.today().year,
        frozen=True,
        strict=True,
    )
    yr_renovated: int = Field(
        ...,
        description="Year the house was renovated (0 if never renovated)",
        examples=[0, 1995, 2010],
        ge=0,
        le=date.today().year,
        frozen=True,
        strict=True,
    )
    zipcode: str = Field(
        ...,
        description="Zip code of the property",
        examples=["98103", "98052"],
        min_length=5,
        max_length=5,
        pattern=r"^\d{5}$",
        frozen=True,
        strict=True,
    )
    lat: float = Field(
        ...,
        description="Latitude coordinate of the property",
        examples=[47.5322],
        ge=-90.0,
        le=90.0,
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )
    long: float = Field(
        ...,
        description="Longitude coordinate of the property",
        examples=[-122.5190],
        ge=-180.0,
        le=180.0,
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )
    sqft_living15: int = Field(
        ...,
        description="Square footage of living space (15 nearest neighbors)",
        examples=[2500],
        gt=0,
        frozen=True,
        strict=True,
    )
    sqft_lot15: int = Field(
        ...,
        description="Square footage of lot (15 nearest neighbors)",
        examples=[6000],
        gt=0,
        frozen=True,
        strict=True,
    )


class HousePricePredictResponse(BaseModel):
    """Response model for real estate price predictions."""

    price: float = Field(
        ...,
        description="Predicted house price in USD",
        ge=0.0,
        frozen=True,
        strict=True,
    )
    model_name: str = Field(
        ...,
        description="Name of the model used",
        examples=["house_price_model_v1"],
        min_length=1,
        frozen=True,
        strict=True,
    )
    model_version: str = Field(
        ...,
        description="Model identifier for versioning",
        examples=["updated_2024-06-01T12:00:00Z"],
        min_length=1,
        frozen=True,
        strict=True,
    )
    zipcode: str = Field(
        ...,
        description="ZIP code used for prediction",
        examples=["98103", "98052"],
        min_length=5,
        max_length=5,
        pattern=r"^\d{5}$",
        frozen=True,
        strict=True,
    )
    demographics_found: bool = Field(
        ...,
        description="Whether demographic data was found for zipcode",
        examples=[True, False],
        frozen=True,
        strict=True,
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Service status",
        examples=["healthy", "unhealthy"],
        frozen=True,
    )
    error: str | None = Field(
        None,
        description="Error message if unhealthy",
        examples=["Model file not found."],
        frozen=True,
        strict=True,
    )


class ModelInfoResponse(BaseModel):
    """Model information and metadata response."""

    model_name: str = Field(
        ...,
        description="Name of the model",
        examples=["house_price_model_v1"],
        min_length=1,
        frozen=True,
        strict=True,
    )
    trained_at: str | None = Field(
        ...,
        description="Date when the model was trained",
        examples=["2025-12-01T01:45:05.916737+00:00"],
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(\+\d{2}:\d{2}|Z)?$",
        frozen=True,
        strict=True,
    )
    model_type: str | list[str | list] = Field(
        ...,
        description="Type of the model",
        examples=["KNeighborsRegressor"],
        min_length=1,
        frozen=True,
    )
    n_features: int = Field(
        ...,
        description="Number of features used by the model",
        examples=[16],
        ge=1,
        frozen=True,
        strict=True,
    )
    features: list[str] = Field(
        ...,
        description="List of feature names used by the model",
        examples=[["bedrooms", "bathrooms", "sqft_living"]],
        min_length=1,
        frozen=True,
        strict=True,
    )
    evaluation_metrics: dict[str, float] = Field(
        {},
        description="Evaluation metrics for the model",
        # examples=[{"rmse": 75000.0, "mae": 50000.0}],
        min_length=1,
        frozen=True,
        strict=True,
    )
    cross_validation_metrics: dict[str, float] = Field(
        {},
        description="Cross-validation metrics for the model",
        # examples=[{"cv_rmse_mean": 80000.0, "cv_rmse_std": 5000.0}],
        frozen=True,
        strict=True,
    )
