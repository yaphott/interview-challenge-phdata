from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from typing import Any


def format_dataframe(df: pd.DataFrame) -> str:
    msg = f"Shape: {df.shape}\n"
    msg += f"dtypes: {df.dtypes.to_dict()}\n"
    msg += f"Columns: {list(df.columns)}\n"
    for col_idx, col in enumerate(df.columns):
        msg += f"Column {col_idx}: {col!r}\n"
        msg += f"    size: {df[col].size}\n"
        msg += f"    dtype: {df[col].dtype}\n"
        msg += f"    Counts:\n"
        msg += f"        NaN: {df[col].isna().sum()}\n"
        msg += f"        unique: {df[col].nunique()}\n"

        if pd.api.types.is_numeric_dtype(df[col]):
            msg += f"        zero: {df[col][df[col] == 0].count()}\n"
            msg += f"        negative: {df[col][df[col] < 0].count()}\n"
            msg += f"        inf: {df[col][df[col] == np.inf].count()}\n"
            msg += f"        -inf: {df[col][df[col] == -np.inf].count()}\n"

        if pd.api.types.is_numeric_dtype(df[col]):
            msg += f"    Statistics:\n"
            msg += f"        min: {df[col].min()}\n"
            msg += f"        p25: {df[col].quantile(0.25)}\n"
            msg += f"        p50: {df[col].quantile(0.50)}\n"
            msg += f"        p75: {df[col].quantile(0.75)}\n"
            msg += f"        p90: {df[col].quantile(0.90)}\n"
            msg += f"        p95: {df[col].quantile(0.95)}\n"
            msg += f"        p99: {df[col].quantile(0.99)}\n"
            msg += f"        max: {df[col].max()}\n"
            msg += f"        mean: {df[col].mean()}\n"
            msg += f"        stddev: {df[col].std()}\n"
            msg += f"        var: {df[col].var()}\n"
            msg += f"        skew: {df[col].skew()}\n"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            msg += f"    Statistics:\n"
            msg += f"        min: {df[col].min()}\n"
            msg += f"        max: {df[col].max()}\n"

        msg += "    value counts:\n"
        value_counts = df[col].value_counts()
        msg += (
            "        "
            + value_counts.to_string(header=False).replace("\n", "\n        ")
            + "\n"
        )

    return msg


def format_dataframe_dict(df: pd.DataFrame) -> dict:
    metrics = {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        # "columns": list(df.columns),
        "columns_info": {},
    }
    for col in df.columns:
        col_info = {
            "size": df[col].size,
            "dtype": str(df[col].dtype),
            "counts": {
                "unique": int(df[col].nunique()),
                "na": int(df[col].isna().sum()),
            },
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["counts"].update(
                {
                    "zero": int(df[col][df[col] == 0].count()),
                    "neg": int(df[col][df[col] < 0].count()),
                    "inf": int(df[col][df[col] == np.inf].count()),
                    "neg_inf": int(df[col][df[col] == -np.inf].count()),
                }
            )
            col_info["statistics"] = {
                "min": df[col].min().item(),
                "p25": df[col].quantile(0.25),
                "p50": df[col].quantile(0.50),
                "p75": df[col].quantile(0.75),
                "p90": df[col].quantile(0.90),
                "p95": df[col].quantile(0.95),
                "p99": df[col].quantile(0.99),
                "max": df[col].max().item(),
                "mean": df[col].mean(),
                "stddev": df[col].std(),
                "var": df[col].var(),
                "skew": df[col].skew(),
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["statistics"] = {
                "min": df[col].min().item(),
                "max": df[col].max().item(),
            }

        value_counts = df[col].value_counts()
        col_info["value_counts"] = {
            str(index): int(count)
            for index, count in zip(value_counts.index, value_counts)
        }

        metrics["columns_info"][col] = col_info

    return metrics


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics_dict = {
        "train_rmse": np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)).item(),
        "test_rmse": np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)).item(),
        "train_mae": metrics.mean_absolute_error(y_train, y_train_pred),
        "test_mae": metrics.mean_absolute_error(y_test, y_test_pred),
        "train_r2": metrics.r2_score(y_train, y_train_pred),
        "test_r2": metrics.r2_score(y_test, y_test_pred),
        "train_mape": (
            np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        ).item(),
        "test_mape": (np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100).item(),
    }
    return metrics_dict


def cross_validate_model(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, Any]:
    # Negative MSE because sklearn uses score maximization
    cv_scores = model_selection.cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = np.sqrt(-cv_scores)
    cv_r2 = model_selection.cross_val_score(model, x, y, cv=cv, scoring="r2", n_jobs=-1)
    return {
        "cv_rmse_mean": cv_rmse.mean().item(),
        "cv_rmse_std": cv_rmse.std().item(),
        "cv_r2_mean": cv_r2.mean().item(),
        "cv_r2_std": cv_r2.std().item(),
    }


def format_model_name(model: BaseEstimator) -> str | list[str | list]:
    if hasattr(model, "steps"):
        components = []
        for _, step in model.steps:
            components.append(format_model_name(step))
        return components
    else:
        return type(model).__name__
