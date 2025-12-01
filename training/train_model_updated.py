#!/usr/bin/env python3
import json
import logging
import pandas as pd
import pathlib
import pickle
from datetime import datetime, timezone
from sklearn import ensemble, model_selection

from training.helpers import (
    create_model_version,
    cross_validate_model,
    evaluate_model,
    format_model_name,
)

PROJECT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent

logger: logging.Logger = logging.getLogger(__name__)


SALES_COLUMNS: list[str] = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "zipcode",
]
DEMOGRAPHICS_COLUMNS: list[str] = [
    "ppltn_qty",
    "urbn_ppltn_qty",
    "sbrbn_ppltn_qty",
    "farm_ppltn_qty",
    "non_farm_qty",
    "medn_hshld_incm_amt",
    "medn_incm_per_prsn_amt",
    "hous_val_amt",
    "edctn_less_than_9_qty",
    "edctn_9_12_qty",
    "edctn_high_schl_qty",
    "edctn_some_clg_qty",
    "edctn_assoc_dgre_qty",
    "edctn_bchlr_dgre_qty",
    "edctn_prfsnl_qty",
    "per_urbn",
    "per_sbrbn",
    "per_farm",
    "per_non_farm",
    "per_less_than_9",
    "per_9_to_12",
    "per_hsd",
    "per_some_clg",
    "per_assoc",
    "per_bchlr",
    "per_prfsnl",
    "zipcode",
]


def create_features(merged_df: pd.DataFrame, dt: datetime) -> pd.DataFrame:
    df_copy = merged_df.copy()
    df_copy["house_age"] = dt.year - df_copy["yr_built"]
    df_copy["was_renovated"] = (df_copy["yr_renovated"] > 0).astype(int)
    df_copy["years_since_renovation"] = dt.year - df_copy["yr_renovated"]
    df_copy.loc[df_copy["yr_renovated"] == 0, "years_since_renovation"] = 0

    df_copy["has_basement"] = (df_copy["sqft_basement"] > 0).astype(int)
    df_copy["large_lot"] = (df_copy["sqft_lot"] > 10000).astype(int)

    return df_copy


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    start_dt = datetime.now(tz=timezone.utc)

    sales_path = PROJECT_DIR / "data" / "kc_house_data.csv"
    if not sales_path.is_file():
        raise FileNotFoundError(f"Sales data file not found: {sales_path}")
    demographics_path = PROJECT_DIR / "data" / "zipcode_demographics.csv"
    if not demographics_path.is_file():
        raise FileNotFoundError(
            f"Demographics data file not found: {demographics_path}"
        )
    future_examples_path = PROJECT_DIR / "data" / "future_unseen_examples.csv"
    if not future_examples_path.is_file():
        raise FileNotFoundError(
            f"Future examples data file not found: {future_examples_path}"
        )
    features_path = (
        PROJECT_DIR
        / "models"
        / f"updated_{start_dt.strftime('%Y-%m-%d_%H-%M-%S')}_model_features.json"
    )
    if features_path.is_file():
        raise FileExistsError(f"Features file already exists: {features_path}")
    metadata_path = (
        PROJECT_DIR
        / "models"
        / f"updated_{start_dt.strftime('%Y-%m-%d_%H-%M-%S')}_model_metadata.json"
    )
    if metadata_path.is_file():
        raise FileExistsError(f"Metadata file already exists: {metadata_path}")
    model_path = (
        PROJECT_DIR
        / "models"
        / f"updated_{start_dt.strftime('%Y-%m-%d_%H-%M-%S')}_model.pkl"
    )
    if not model_path.parent.is_dir():
        model_path.parent.mkdir()

    sales_df = pd.read_csv(
        sales_path,
        usecols=SALES_COLUMNS,
        dtype={"zipcode": "string"},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug("Sales data columns: %s", sales_df.columns.tolist())

    demographics_df = pd.read_csv(
        demographics_path,
        usecols=DEMOGRAPHICS_COLUMNS,
        dtype={"zipcode": "string"},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug("Demographics data columns: %s", demographics_df.columns.tolist())

    merged_df = pd.merge(
        sales_df,
        demographics_df,
        how="inner",
        on="zipcode",
    )
    logger.debug("Merged data columns: %s", merged_df.columns.tolist())

    features_df = create_features(merged_df, start_dt)

    y = features_df.pop("price")
    x = features_df
    print(f"{y.shape=}, {x.shape=}")
    print(f"{x.columns=}")

    future_examples_df = pd.read_csv(
        future_examples_path,
        dtype={"zipcode": str},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug(
        "Future examples data columns: %s", future_examples_df.columns.tolist()
    )

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    logger.debug("Training model")
    model = ensemble.GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    model.fit(x_train, y_train)

    eval_metrics = evaluate_model(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    logger.info("Evaluation metrics: %s", eval_metrics)
    cv_metrics = cross_validate_model(model, x, y)
    logger.info("Cross-validation metrics: %s", cv_metrics)

    model_name = "updated_model"
    metadata = {
        "model_name": model_name,
        "model_version": create_model_version(model_name, start_dt),
        "trained_at": start_dt.isoformat(),
        "model_type": format_model_name(model),
        "n_features": x.shape[1],
        "features": x.columns.tolist(),
        "evaluation_metrics": eval_metrics,
        "cross_validation_metrics": cv_metrics,
    }

    logger.info("Writing model to %s", model_path)
    with open(model_path, mode="wb") as f:
        pickle.dump(model, f)
    logger.info("Writing features to %s", features_path)
    with open(features_path, mode="w", encoding="utf-8") as f:
        json.dump(list(x.columns), f, indent=4)
    logger.info("Writing metadata to %s", metadata_path)
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    main()
