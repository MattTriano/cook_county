import os

import pandas as pd
from pandas.api.types import CategoricalDtype

from utils import (
    extract_file_from_url,
    get_project_root_dir,
    standardize_mistakenly_int_parsed_categorical_series,
    standardize_and_zerofill_intlike_values_to_str,
)


def load_raw_cook_county_property_locations(
    root_dir: os.path = get_project_root_dir(), force_repull: bool = False
) -> pd.DataFrame:
    property_locations_df = extract_file_from_url(
        file_path=os.path.join(root_dir, "data_raw", "cook_county_property_locations.csv"),
        url="https://datacatalog.cookcountyil.gov/api/views/c49d-89sn/rows.csv?accessType=DOWNLOAD",
        data_format="csv",
        force_repull=force_repull,
        return_df=True,
    )
    return property_locations_df


def standardize_and_zerofill_intlike_cook_county_property_location_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["pin"] = standardize_and_zerofill_intlike_values_to_str(series=df["pin"], zerofill=14)
    df["puma"] = standardize_and_zerofill_intlike_values_to_str(series=df["puma"], zerofill=4)
    df["nbhd"] = standardize_and_zerofill_intlike_values_to_str(series=df["nbhd"], zerofill=3)
    df["reps_dist"] = standardize_and_zerofill_intlike_values_to_str(
        series=df["reps_dist"], zerofill=2
    )
    df["senate_dist"] = standardize_and_zerofill_intlike_values_to_str(
        series=df["senate_dist"], zerofill=2
    )
    df["tif_agencynum"] = standardize_and_zerofill_intlike_values_to_str(
        series=df["tif_agencynum"], zerofill=2
    )
    df["township"] = standardize_and_zerofill_intlike_values_to_str(
        series=df["township"], zerofill=2
    )
    df["ward"] = standardize_and_zerofill_intlike_values_to_str(series=df["ward"], zerofill=2)
    return df


def typeset_ordered_categorical_fs_flood_factor_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """The property's First Street Flood Factor, a numeric integer from 1-10
    (where 1 = minimal and 10 = extreme) based on flooding risk to the
    building footprint. Flood risk is defined as a combination of cumulative
    risk over 30 years and flood depth. Flood depth is calculated at the
    lowest elevation of the building footprint (large."""
    flood_factor_risk_categories = CategoricalDtype(
        categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ordered=True
    )
    df["fs_flood_factor"] = df["fs_flood_factor"].astype(flood_factor_risk_categories)
    return df


def typeset_ordered_categorical_fs_flood_risk_direction_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """The property's flood risk direction represented in a numeric value
    based on the change in risk for the location from 2020 to 2050 for the
    climate model realization of the RCP 4.5 mid emissions scenario.
    -1 = descreasing, 0 = stationary, 1 = increasing.
    Data provided by First Street and academics at UPenn."""
    flood_risk_map = {
        -1: "risk_decreasing",
        0: "risk_stationary",
        1: "risk_increasing",
    }
    categories = list(flood_risk_map.values())
    if any([value not in df["fs_flood_risk_direction"].unique() for value in categories]):
        df["fs_flood_risk_direction"] = df["fs_flood_risk_direction"].map(flood_risk_map)
    ordered_cats = CategoricalDtype(categories=categories, ordered=True)
    df["fs_flood_risk_direction"] = df["fs_flood_risk_direction"].astype(ordered_cats)
    return df


def transform_cook_county_property_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_and_zerofill_intlike_cook_county_property_location_columns(df=df)
    df = typeset_simple_boolean_columns(
        df=df,
        boolean_columns=[
            "withinmr100",
            "withinmr101300",
            "ohare_noise",
            "floodplain",
            "indicator_has_address",
            "indicator_has_latlon",
        ],
    )
    property_locations_df = typeset_simple_category_columns(
        df=df,
        category_columns=[
            "commissioner_dist",
            "senate_dist",
            "township_name",
            "township",
            "puma",
            "ward",
            "ssa_no",
            "ssa_name",
            "reps_dist",
            "mailing_state",
            "school_hs_district",
            "municipality",
            "municipality_fips",
            "property_city",
            "nbhd",
            "tif_agencynum",
            "school_elem_district",
            "tract_geoid",
            "mailing_city",
            "mailing_zip",
            "property_zip",
        ],
    )
    df = typeset_ordered_categorical_fs_flood_factor_feature(df=df)
    df = typeset_ordered_categorical_fs_flood_risk_direction_feature(df=df)
    return df
