"""
Configuration management for the API
"""

import os
from pathlib import Path

LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()

BASE_DIR: Path = Path(__file__).parent.parent

MODEL_PATH_STR: str = os.getenv("MODEL_PATH", "")
MODEL_PATH: Path = (
    Path(MODEL_PATH_STR) if MODEL_PATH_STR else BASE_DIR / "model" / "model.pkl"
)

DEMOGRAPHICS_PATH_STR: str = os.getenv("DEMOGRAPHICS_PATH", "")
DEMOGRAPHICS_PATH: Path = (
    Path(DEMOGRAPHICS_PATH_STR)
    if DEMOGRAPHICS_PATH_STR
    else BASE_DIR / "data" / "zipcode_demographics.csv"
)

API_TITLE: str = "Sound Realty Home Price Prediction API"
API_DESCRIPTION: str = """
This API provides home price predictions for properties in the Seattle area.

## Features
- Predict home prices based on property attributes
- Automatic demographic data enrichment by zipcode
- Multiple prediction endpoints (full features and minimal features)
- Model metadata and health check endpoints

## Usage
Submit property details via POST requests to get price predictions.
"""
API_VERSION: str = "1.0.0"

# Server settings
HOST: str = os.getenv("API_HOST", "0.0.0.0")
PORT: int = int(os.getenv("API_PORT", "8000"))
WORKERS: int = int(os.getenv("API_WORKERS", "4"))
