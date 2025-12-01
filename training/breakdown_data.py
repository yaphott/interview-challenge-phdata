#!/usr/bin/env python3
import json
import logging
import pandas as pd
import pathlib

from training.helpers import format_dataframe_dict

PROJECT_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    sales_path = PROJECT_DIR / "data" / "kc_house_data.csv"
    if not sales_path.is_file():
        raise FileNotFoundError(f"Sales data file not found: {sales_path}")
    sales_breakdown_path = sales_path.with_stem(
        sales_path.stem + "_breakdown"
    ).with_suffix(".json")

    demographics_path = PROJECT_DIR / "data" / "zipcode_demographics.csv"
    if not demographics_path.is_file():
        raise FileNotFoundError(
            f"Demographics data file not found: {demographics_path}"
        )
    demographics_breakdown_path = demographics_path.with_stem(
        demographics_path.stem + "_breakdown"
    ).with_suffix(".json")

    future_examples_path = PROJECT_DIR / "data" / "future_unseen_examples.csv"
    if not future_examples_path.is_file():
        raise FileNotFoundError(
            f"Future examples data file not found: {future_examples_path}"
        )
    future_examples_breakdown_path = future_examples_path.with_stem(
        future_examples_path.stem + "_breakdown"
    ).with_suffix(".json")

    sales_df = pd.read_csv(
        sales_path,
        dtype={"zipcode": "string"},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug("Writing sales data breakdown to %s", sales_breakdown_path)
    with open(sales_breakdown_path, mode="w", encoding="utf-8") as f:
        json.dump(format_dataframe_dict(sales_df), f, indent=4)

    demographics_df = pd.read_csv(
        demographics_path,
        dtype={"zipcode": "string"},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug(
        "Writing demographics data breakdown to %s", demographics_breakdown_path
    )
    with open(demographics_breakdown_path, mode="w", encoding="utf-8") as f:
        json.dump(format_dataframe_dict(demographics_df), f, indent=4)

    future_examples_df = pd.read_csv(
        future_examples_path,
        dtype={"zipcode": "string"},
        low_memory=False,
        on_bad_lines="error",
    )
    logger.debug(
        "Writing future examples data breakdown to %s", future_examples_breakdown_path
    )
    with open(future_examples_breakdown_path, mode="w", encoding="utf-8") as f:
        json.dump(format_dataframe_dict(future_examples_df), f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)04d [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    main()
