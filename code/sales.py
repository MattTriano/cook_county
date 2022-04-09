import os

import pandas as pd

from utils import (
    extract_file_from_url,
    get_project_root_dir,
    transform_date_columns,
    typeset_simple_boolean_columns,
    typeset_simple_category_columns,
    typeset_ordered_categorical_feature,
    standardize_column_names,
    standardize_mistakenly_int_parsed_categorical_series,
    standardize_and_zerofill_intlike_values_to_str,
)


def load_raw_cook_county_property_sales(
    root_dir: os.path = get_project_root_dir(), force_repull: bool = False
) -> pd.DataFrame:
    df = extract_file_from_url(
        file_path=os.path.join(root_dir, "data_raw", "cook_county_property_sales.csv"),
        url="https://datacatalog.cookcountyil.gov/api/views/93st-4bxh/rows.csv?accessType=DOWNLOAD",
        data_format="csv",
        force_repull=force_repull,
        return_df=True,
    )
    return df


def preprocess_pin_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df["pin"] = df["pin"].str.replace("-", "")
    return df


def preprocess_property_sales_arms_length_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Indicator for whether the CCAO believes this is an arms' length transaction."""
    arms_length_map = {
        0: "no",
        1: "yes",
        9: "unknown",
    }
    categories = list(arms_length_map.values())
    if any([cat not in df["arms_length"].unique() for cat in categories]):
        df["arms_length"] = df["arms_length"].map(arms_length_map)
    df["arms_length"] = df["arms_length"].astype("category")
    return df


def preprocess_property_sales_deed_type_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    deed_type_map = {
        "W": "Warranty",
        "O": "Other",
        "o": "Other",
        "T": "Trustee",
        "Y": "Trustee",
    }
    categories = list(deed_type_map.values())
    if any([cat not in df["deed_type"].unique() for cat in categories]):
        df["deed_type"] = df["deed_type"].map(deed_type_map)
    df["deed_type"] = df["deed_type"].astype("category")
    return df


def transform_cook_county_property_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df=df)
    df = preprocess_pin_numbers(df=df)
    df = transform_date_columns(df=df, date_cols=["recorded_date", "executed_date"])
    df = preprocess_property_sales_arms_length_values(df=df)
    df = preprocess_property_sales_deed_type_values(df=df)
    df["year"] = typeset_ordered_categorical_feature(series=df["year"])
    return df
