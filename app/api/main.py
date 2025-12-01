#!/usr/bin/env python3
from __future__ import annotations

import logging
import time
from api import config
from api.models import (
    HealthResponse,
    HouseFeatures,
    HousePricePredictResponse,
    ModelInfoResponse,
)
from api.predictor import get_predictor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import ORJSONResponse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from starlette.requests import Request


logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=config.LOG_LEVEL_STR,
    format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        _ = get_predictor(
            model_path=config.MODEL_PATH,
            demographics_path=config.DEMOGRAPHICS_PATH,
        )
    except Exception as err:
        logger.error("Failed to load predictor: %s", err)
        raise err
    yield
    logger.info("Shutting down application")


app: FastAPI = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    logger.info(
        "Received requests: %s",
        {k: v for k, v in dict(request).items() if k != "app"}
        | {"body": await request.body()},
    )  # This should all really be stored as fields in JSON logging or spanned traces
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/", tags=["General"])
async def root() -> dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "message": "Sound Realty Home Price Prediction API",
        "version": config.API_VERSION,
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Health check report.",
            },
            "/docs": {
                "method": "GET",
                "description": "API documentation (Swagger UI).",
            },
            "/predict": {
                "method": "POST",
                "description": "Predict home price using full house features.",
            },
            # "/predict/minimal": {
            #     "method": "POST",
            #     "description": "Predict home price using minimal house features.",
            # },
            "/model/info": {
                "method": "GET",
                "description": "Model information and metadata.",
            },
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health() -> HealthResponse | ORJSONResponse:
    """Check the health of the API service.

    Returns:
        HealthResponse: Health status of the API service.
    """
    try:
        _ = get_predictor(
            model_path=config.MODEL_PATH,
            demographics_path=config.DEMOGRAPHICS_PATH,
        )
        return HealthResponse(status="healthy", error=None)
    except Exception as err:
        logger.error("Health check failed: %s", err)
        return ORJSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(err)},
        )


@app.post("/predict", response_model=HousePricePredictResponse, tags=["Prediction"])
async def predict(house: HouseFeatures) -> HousePricePredictResponse:
    """Predict home price using full house features.

    Args:
        house (HouseFeatures): House features used for the real estate price prediction.

    Returns:
        PricePredictResponse: Response containing the predicted price.
    """
    try:
        predictor = get_predictor(
            model_path=config.MODEL_PATH,
            demographics_path=config.DEMOGRAPHICS_PATH,
        )
        data = house.model_dump()
        result = predictor.predict(data, minimal=False)
        response = HousePricePredictResponse(**result)
        return response
    except Exception as err:
        logger.error("Failed to get predictor: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err


# @app.post("/predict/minimal", response_model=HousePricePredictResponse, tags=["Prediction"])
# async def predict_minimal(house: MinimalHouseFeatures) -> HousePricePredictResponse:
#     """Predict home price using minimal house features.
#
#     Args:
#         house (MinimalHouseFeatures): House features used for the real estate price prediction.
#
#     Returns:
#         PricePredictResponse: Response containing the predicted price.
#     """
#     try:
#         predictor = get_predictor(
#             model_path=config.MODEL_PATH,
#             demographics_path=config.DEMOGRAPHICS_PATH,
#         )
#         data = house.model_dump()
#         result = predictor.predict(data, minimal=True)
#         response = HousePricePredictResponse(**result)
#         return response
#     except Exception as err:
#         logger.error("Failed to get predictor: %s", err)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error",
#         ) from err


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get model information and metadata.

    Returns:
        ModelInfoResponse: Model metadata including feature names and other relevant information.
    """
    try:
        predictor = get_predictor(
            model_path=config.MODEL_PATH,
            demographics_path=config.DEMOGRAPHICS_PATH,
        )
        info = ModelInfoResponse(**predictor.metadata)
        return info
    except Exception as err:
        logger.error("Failed to get model info: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err


# def main() -> None:
#     logging.basicConfig(
#         level=LOG_LEVEL_STR,
#         format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
#     )
#     import uvicorn
#
#     config = uvicorn.Config("main:app", host="127.0.0.1", port=8000, log_level=LOG_LEVEL_STR.lower())
#     server = uvicorn.Server(config)
#     server.run()
#
#
# if __name__ == "__main__":
#     main()
