from __future__ import annotations

import copy
import json
import logging
import pandas as pd
import pickle
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from sklearn.base import BaseEstimator
    from typing import Any

logger: logging.Logger = logging.getLogger(__name__)
_predictors: dict[tuple[str, ...], HousePricePredictor] = {}


class HousePricePredictor:
    def __init__(
        self,
        model_path: Path,
        demographics_path: Path,
    ) -> None:
        self._model_path: Path = model_path
        self._demographics_path: Path = demographics_path

        self._model: BaseEstimator
        self._features: list[str]
        self._metadata: dict[str, Any]
        self._load_model()

        self._demographics_df: pd.DataFrame
        self._load_demographics()

    def _load_model(self) -> None:
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        features_path = self._model_path.with_stem(
            self._model_path.stem + "_features"
        ).with_suffix(".json")
        if not features_path.is_file():
            raise FileNotFoundError(f"Model features file not found: {features_path}")
        metadata_path = self._model_path.with_stem(
            self._model_path.stem + "_metadata"
        ).with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")
        try:
            logger.info("Loading model from %s", self._model_path)
            with open(self._model_path, mode="rb") as f:
                self._model = pickle.load(f)
            logger.debug("Loading model features from %s", features_path)
            with open(features_path, mode="r", encoding="utf-8") as f:
                self._features = json.load(f)
            logger.debug("Loading model metadata from %s", metadata_path)
            with open(metadata_path, mode="r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        except Exception as err:
            logger.error("Failed to load model from %s: %s", self._model_path, err)
            raise RuntimeError(
                f"Failed to load model from {self._model_path}: {err}"
            ) from err

    def _load_demographics(self) -> None:
        if not self._demographics_path.is_file():
            raise FileNotFoundError(
                f"Demographics file not found: {self._demographics_path}"
            )
        try:
            logger.info("Loading demographics data from %s", self._demographics_path)
            self._demographics_df = pd.read_csv(
                self._demographics_path,
                dtype={"zipcode": "string"},
                low_memory=False,
                on_bad_lines="error",
            )
            self._demographics_df = self._demographics_df.set_index("zipcode")
        except Exception as err:
            logger.error(
                "Failed to load demographics data from %s: %s",
                self._demographics_path,
                err,
            )
            raise RuntimeError(
                f"Failed to load demographics data from {self._demographics_path}: {err}"
            ) from err

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def demographics_path(self) -> Path:
        return self._demographics_path

    @property
    def features(self) -> list[str]:
        return self._features.copy()

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def predict(self, data: dict[str, Any], minimal: bool = False) -> dict[str, Any]:
        now_dt = datetime.now(tz=timezone.utc)
        # if minimal:
        #     input_data = self._fill_missing_features(data)

        demo_data, demo_found = self._get_demographic_data(data["zipcode"])
        if not demo_found:
            logger.warning(
                "Demographic data not found for zipcode: %s", data["zipcode"]
            )

        features = self._create_features(data | demo_data, now_dt)
        features_df = pd.DataFrame([features], columns=self._features)
        if features_df.isna().any().any():
            missing_cols = features_df.columns[features_df.isna().any()].tolist()
            logger.error("Missing feature values for columns: %s", missing_cols)
            raise ValueError(f"Missing feature values for columns: {missing_cols}")

        pred = self._model.predict(features_df)[0]
        return {
            "price": pred,
            "model_name": self._metadata.get("model_name", "Unknown"),
            "model_version": self._metadata.get("model_version", "Unknown"),
            "zipcode": data["zipcode"],
            "demographics_found": demo_found,
        }

    def _get_demographic_data(self, zipcode: str) -> tuple[dict[str, Any], bool]:
        """Get demographic information for the provided zipcode.

        Args:
            zipcode (str): Zipcode to query.

        Returns:
            tuple[dict[str, Any], bool]: Demographic data and flag indicating if found.
        """
        demo_data: dict[str, Any]
        found = False
        if zipcode in self._demographics_df.index:
            demo_data = self._demographics_df.loc[zipcode].to_dict()
            found = True
        else:
            demo_data = self._demographics_df.median(numeric_only=True).to_dict()
        return demo_data, found

    def _create_features(self, data: dict[str, Any], dt: datetime) -> list[Any]:
        """Engineer features from the input data.

        Args:
            data (dict[str, Any]): Original feature data.
            dt (datetime): Current datetime for time-based features.

        Returns:
            list[Any]: Prepared feature data for prediction.
        """
        features = {}

        # House age
        features["house_age"] = dt.year - data["yr_built"]

        # Was renovated
        features["was_renovated"] = 1 if data["yr_renovated"] > 0 else 0

        # Years since renovation
        if data["yr_renovated"] > 0:
            features["years_since_renovation"] = dt.year - data["yr_renovated"]
        else:
            features["years_since_renovation"] = 0

        # Has basement
        features["has_basement"] = 1 if data["sqft_basement"] > 0 else 0

        # Large lot
        features["large_lot"] = 1 if data["sqft_lot"] > 10000 else 0

        for feat in self._features:
            if feat in data:
                if feat in features:
                    raise ValueError(
                        f"Feature {feat!r} already set in engineered features"
                    )
                features[feat] = data[feat]
            elif feat not in features:
                raise ValueError(f"Missing feature {feat!r} in input data")

        return [features[feat] for feat in self._features if feat in self._features]


def get_predictor(model_path: Path, **data_paths) -> HousePricePredictor:
    """Get or instantiate the cached predictor.

    Args:
        model_path (Path): Path to the model file.
        data_paths (dict[str, Path]): Additional data file paths.

    Returns:
        Predictor: Global instance of predictor.
    """
    global _predictors
    kwargs: dict[str, Path] = {"model_path": model_path} | data_paths
    key = tuple(str(kwargs[k].resolve()) for k in sorted(kwargs))
    if key not in _predictors:
        _predictors[key] = HousePricePredictor(**kwargs)
    return _predictors[key]
