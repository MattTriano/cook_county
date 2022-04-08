import os

import pandas as pd

from utils import (
    extract_file_from_url
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