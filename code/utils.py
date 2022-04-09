from datetime import datetime
import os
import re
from typing import Dict, List, Union, Optional
from urllib.request import urlretrieve

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_bool_dtype, is_bool
import requests
from shapely.geometry import Point


def get_df_column_details(df: pd.DataFrame) -> pd.DataFrame:
    col_list = list(df.columns)
    n_rows = df.shape[0]
    df_details = pd.DataFrame(
        {
            "feature": [col for col in col_list],
            "unique_vals": [df[col].nunique() for col in col_list],
            "pct_unique": [round(100 * df[col].nunique() / n_rows, 4) for col in col_list],
            "null_vals": [df[col].isnull().sum() for col in col_list],
            "pct_null": [round(100 * df[col].isnull().sum() / n_rows, 4) for col in col_list],
        }
    )
    df_details = df_details.sort_values(by="unique_vals")
    df_details = df_details.reset_index(drop=True)
    return df_details


def get_project_root_dir() -> os.path:
    if "__file__" in globals().keys():
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    else:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(".")))
    # assert ".git" in os.listdir(root_dir)
    return root_dir


def setup_project_structure(project_root_dir: os.path = get_project_root_dir()) -> None:
    os.makedirs(os.path.join(project_root_dir, "data_raw"), exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "data_intermediate"), exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "data_clean"), exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "code"), exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "output"), exist_ok=True)


def extract_csv_from_url(
    file_path: os.path, url: str, force_repull: bool = False, return_df: bool = True
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path) or force_repull:
        urlretrieve(url, file_path)
    if return_df:
        return pd.read_csv(file_path)


def extract_file_from_url(
    file_path: os.path,
    url: str,
    data_format: str,
    force_repull: bool = False,
    return_df: bool = True,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path) or force_repull:
        urlretrieve(url, file_path)
    if return_df:
        if data_format in ["csv", "zipped_csv"]:
            return pd.read_csv(file_path)
        elif data_format in ["shp", "geojson"]:
            return gpd.read_file(file_path)


def make_point_geometry(df: pd.DataFrame, long_col: str, lat_col: str) -> pd.Series:
    latlong_df = df[[long_col, lat_col]].copy()
    df["geometry"] = pd.Series(map(Point, latlong_df[long_col], latlong_df[lat_col]))
    return df


def geospatialize_df_with_point_geometries(
    df: pd.DataFrame, long_col: str, lat_col: str, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    df = df.copy()
    gdf = make_point_geometry(df=df, long_col=long_col, lat_col=lat_col)
    gdf = gpd.GeoDataFrame(gdf, crs=crs)
    return gdf


def typeset_datetime_column(dt_series: pd.Series, dt_format: Optional[str]) -> pd.Series:
    dt_series = dt_series.copy()
    if not is_datetime64_any_dtype(dt_series):
        if dt_format is not None:
            try:
                dt_series = pd.to_datetime(dt_series, format=dt_format)
            except:
                dt_series = pd.to_datetime(dt_series)
        else:
            dt_series = pd.to_datetime(dt_series)
    return dt_series


def drop_columns(df: pd.DataFrame, columns_to_drop: List) -> pd.DataFrame:
    assert all(
        [col in df.columns for col in columns_to_drop]
    ), "columns_to_drop include missing columns"
    df = df.drop(columns=columns_to_drop)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(col.lower().split(" ")) for col in df.columns]
    return df


def standardize_mistakenly_int_parsed_categorical_series(
    series: pd.Series, zerofill: Optional[int] = None
) -> pd.Series:
    if zerofill is not None:
        series = series.astype("Int64").astype("string").str.zfill(zerofill).astype("category")
    else:
        series = series.astype("Int64").astype("string").astype("category")
    return series


def typeset_simple_category_columns(df: pd.DataFrame, category_columns: List[str]) -> pd.DataFrame:
    for category_column in category_columns:
        df[category_column] = df[category_column].astype("category")
    return df


def typeset_ordered_categorical_column(
    series: pd.Series, ordered_category_values: List
) -> pd.Series:
    series = series.astype(CategoricalDtype(categories=ordered_category_values, ordered=True))
    return series


def typeset_ordered_categorical_feature(series: pd.Series) -> pd.Series:
    series = series.copy()
    series_categories = list(series.unique())
    series_categories.sort()
    series = series.astype(CategoricalDtype(categories=series_categories, ordered=True))
    return series


def transform_date_columns(
    df: pd.DataFrame, date_cols: List[str], dt_format: str = "%m/%d/%Y %I:%M:%S %p"
) -> pd.DataFrame:
    for date_col in date_cols:
        df[date_col] = typeset_datetime_column(dt_series=df[date_col], dt_format=dt_format)
    return df


def map_column_to_boolean_values(series: pd.Series, true_values: List[str]) -> pd.DataFrame:
    series = series.copy()
    true_mask = series.isin(true_values)
    if is_bool_dtype(series) or is_bool(series):
        return series
    series.loc[~true_mask] = False
    series.loc[true_mask] = True
    series = series.astype("boolean")
    return series


def typeset_simple_boolean_columns(df: pd.DataFrame, boolean_columns: List[str]) -> pd.DataFrame:
    for boolean_col in boolean_columns:
        df[boolean_col] = map_column_to_boolean_values(series=df[boolean_col], true_values=[1])
    return df


def get_socrata_table_metadata(table_id: str) -> Dict:
    api_call = f"http://api.us.socrata.com/api/catalog/v1?ids={table_id}"
    response = requests.get(api_call)
    if response.status_code == 200:
        response_json = response.json()
        results = {"_id": table_id, "time_of_collection": datetime.utcnow()}
        results.update(response_json["results"][0])
        return results


if __name__ == "__main__":
    root_dir = get_project_root_path()
    print(root_dir)
    print(os.listdir(root_dir))
