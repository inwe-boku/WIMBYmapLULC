import json
import os
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.mask import mask
from scipy.spatial import distance
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from . import PACKAGE_DIR


def read_yaml_config(config_filename: str) -> dict:
    """
    Reads a YAML configuration file and returns the configuration as a dictionary.

    Args:
        config_filename (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data.
    """
    try:
        with open(config_filename, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: File '{config_filename}' not found.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return {}


def turbines_geojson_to_gdf(turbines_geojson: dict) -> gpd.GeoDataFrame:
    """
    Converts a turbines GeoJSON dictionary to a GeoPandas GeoDataFrame.

    Expects a GeoJSON dictionary with a 'features' key containing a list of features.
    Each feature should be a dictionary with 'geometry' and optionally 'properties'.

    Args:
        turbines_geojson (dict): The GeoJSON data as a dictionary.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the turbine data with CRS set to EPSG:4326.
    """
    features = turbines_geojson.get("features", [])
    if not features:
        print("No features found in the GeoJSON data.")
        return gpd.GeoDataFrame(crs="EPSG:4326")

    records = []
    geometries = []
    for feature in features:
        properties = feature.get("properties", {})
        geom = feature.get("geometry", None)
        if geom is not None:
            geometries.append(shape(geom))
        else:
            geometries.append(None)
        records.append(properties)

    # Create the GeoDataFrame with the CRS explicitly set to EPSG:4326
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf


def cluster_distance(points, threshold=9) -> int:
    """
    Computes the average minimum distance among a set of points,
    considering only distances that are greater than or equal to a given threshold.

    Each point in the input is expected to have attributes `x` and `y`.

    Parameters:
        points (iterable): An iterable of objects with attributes `x` and `y`.
        threshold (float, optional): The minimum distance threshold. Distances below
                                     this value will be ignored (set to infinity).
                                     Defaults to 9.

    Returns:
        int: The average of the minimum distances (converted to an integer)
             among the points after filtering distances below the threshold.
    """
    # Extract coordinates from the points
    coordinates = np.array([(point.x, point.y) for point in points])

    # Compute the pairwise Euclidean distance matrix
    dist_matrix = distance.cdist(coordinates, coordinates)

    # Set all distances less than the threshold to infinity
    dist_matrix[dist_matrix < threshold] = np.inf

    # For each point, find the minimum distance (ignoring any distances set to infinity)
    min_distances = np.min(dist_matrix, axis=1)

    # Compute and return the mean of these minimum distances as an integer
    return int(np.mean(min_distances))


def create_hulls(points, hull_type="concave"):
    if hull_type == "buffer":
        hull_gdf = create_buffer_polygons(points)
    elif hull_type == "convex":
        hull_gdf = create_convex_hulls(points)
    elif hull_type == "concave":
        hull_gdf = create_concave_hulls(points)
    else:
        raise ValueError(
            "Invalid hull_type. Choose from 'buffer', 'convex', or 'concave'."
        )
    hull_gdf.reset_index(inplace=True)
    hull_gdf["hull_id"] = hull_gdf.index + 1  # Start IDs from 1
    return hull_gdf


def create_buffer_polygons(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates buffer polygons for each geometry in a GeoDataFrame based on the buffer
    distance provided in the 'buffer' column.

    This function uses GeoPandas' vectorized buffer operation to generate buffer polygons
    for each geometry. It returns a new GeoDataFrame with all original attributes, but
    the 'geometry' column is replaced by the computed buffer polygons.

    Args:
        points (gpd.GeoDataFrame): A GeoDataFrame containing geometries and a 'buffer'
                                   column specifying the buffer distance for each geometry.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with the buffer polygons as its geometry.
    """
    # Compute buffer polygons using the values from the 'buffer' column
    buffer_polygons = points.buffer(points["buffer"].values)

    # Create a shallow copy of the input GeoDataFrame
    buffer_gdf = points.copy(deep=False)

    # Replace the geometry column with the computed buffer polygons
    buffer_gdf["geometry"] = buffer_polygons

    return buffer_gdf


def create_convex_hulls(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates convex hulls for clusters of points by:
      - Dissolving the input GeoDataFrame based on the 'cluster' column.
      - Computing the convex hull for each dissolved group.
      - Buffering the convex hull using the 'buffer' column values.

    The function returns a new GeoDataFrame with its geometry replaced by the buffered convex hulls.

    Parameters:
        points (gpd.GeoDataFrame): Input GeoDataFrame containing:
            - A 'cluster' column for grouping points.
            - A 'buffer' column specifying the buffer distance for each group.
            - A 'geometry' column with the point geometries.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with the updated geometry based on the buffered convex hulls.
    """
    # Dissolve the points by the 'cluster' attribute.
    grouped = points.dissolve()

    # Compute the convex hull for each group.
    convex_hulls = grouped.convex_hull

    # Buffer the convex hulls using the corresponding buffer distances.
    buffered_hulls = convex_hulls.buffer(grouped["buffer"].values)

    # Create a shallow copy of the grouped GeoDataFrame to retain other attributes.
    result_gdf = grouped.copy(deep=False)
    result_gdf.drop(columns=["geometry"], inplace=True)
    result_gdf["geometry"] = buffered_hulls

    return result_gdf


def create_concave_hulls(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates concave hulls for the input GeoDataFrame by dissolving its geometries,
    computing the concave hull on the dissolved data, and then buffering the result
    using the 'buffer' column from the dissolved GeoDataFrame.

    This function assumes that:
      - The input GeoDataFrame has a 'buffer' column with numeric values for buffering.
      - The dissolved GeoDataFrame supports a custom `concave_hull()` method that computes
        a concave hull.

    Parameters:
        points (gpd.GeoDataFrame): A GeoDataFrame containing geometries and a 'buffer' column.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with its geometry replaced by the buffered concave hull.
    """
    # Dissolve the geometries to aggregate them into a single (or grouped) geometry.
    dissolved = points.dissolve()

    # Compute the concave hull on the dissolved geometry.
    # Note: The 'concave_hull' method should be defined for the dissolved GeoDataFrame.
    concave = dissolved.concave_hull()

    # Buffer the computed concave hull using the buffer distances from the dissolved data.
    buffered_concave = concave.buffer(dissolved["buffer"].values)

    # Create a shallow copy of the dissolved GeoDataFrame and update its geometry.
    result_gdf = dissolved.copy(deep=False)
    result_gdf["geometry"] = buffered_concave

    return result_gdf


def sample_raster_values_within_polygon(
    raster_path: str,
    polygon: Union[gpd.GeoDataFrame, gpd.GeoSeries, BaseGeometry],
    result_type: str = "count",
):
    """
    Samples raster values within a given polygon and computes:
      - The pixel count per unique raster value.
      - The average (mean) raster value within the polygon.

    The function ensures that the polygon (or hull) is transformed to the same CRS as the raster file.

    Parameters:
      raster_path (str): Path to the raster file.
      polygon (GeoDataFrame, GeoSeries, or shapely geometry): The polygon geometry for sampling.
          If a GeoPandas object is provided, the first geometry is used after transforming it to the raster's CRS.
      result_type (str): Specifies the output:
          "count" returns a dictionary mapping each unique raster value to its pixel count.
          "mean" returns the average raster value as a float.
          Defaults to "count".

    Returns:
      dict or float:
          dict: {raster_value: count} if result_type is "count" (empty dict if no valid pixels)
          float: average raster value if result_type is "mean" (None if no valid pixels)

    Raises:
      ValueError: If result_type is not "count" or "mean".
    """
    with rasterio.open(raster_path) as src:
        target_crs = src.crs

        # Transform the polygon to the raster's CRS if necessary.
        if isinstance(polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
            if polygon.crs is not None and polygon.crs != target_crs:
                polygon = polygon.to_crs(target_crs)
            # Use the first geometry in the GeoPandas object.
            geom = (
                polygon.geometry.iloc[0]
                if isinstance(polygon, gpd.GeoDataFrame)
                else polygon.iloc[0]
            )
        else:
            # Assume the shapely geometry is already in the correct CRS.
            geom = polygon

        # Convert the geometry to a GeoJSON-like dict.
        geojson = [mapping(geom)]
        out_image, _ = mask(src, geojson, crop=True)
        data = out_image[0]  # Assuming a single-band raster.
        nodata = src.nodata

    # Filter out nodata values if defined.
    valid_data = data if nodata is None else data[data != nodata]

    # Handle the case where no valid sampling values are present.
    if valid_data.size == 0:
        if result_type == "count":
            return {}
        elif result_type == "mean":
            return None
        else:
            raise ValueError("Invalid result_type. Choose 'count' or 'mean'.")

    unique, counts = np.unique(valid_data, return_counts=True)
    pixel_count = {int(k): int(v) for k, v in zip(unique, counts)}
    mean_value = float(np.mean(valid_data))

    if result_type == "count":
        return pixel_count
    elif result_type == "mean":
        return mean_value
    else:
        raise ValueError("Invalid result_type. Choose 'count' or 'mean'.")


def find_best_matching_row(sampledata, csvdata, config):
    """
    Given a sample dictionary and a CSV (as a pandas DataFrame),
    adapt the sample data to match the CSV columns by:
      - Flattening any nested dictionaries for keys starting with "clc" into the main dict
        with keys formatted as "<key>_VAL_<subkey>".
      - Keeping only clc values (in pixels, where each pixel represents 10,000 m²) that are at least
        a certain percentage of the total area, as defined in config["clc_minperc"].
    Then, apply z-score normalization to the matching columns and compute a weighted r_square
    (sum of weighted squared differences) for each row. Weighting factors are specified in
    config["weighting_factors"] (a weight of 0 means the column is ignored).
    Additionally, for all clc columns used in the matching, only rows with non-zero values in these
    columns are considered (rows with any zero in these columns are disqualified).

    If no valid match is found (i.e. all rows are disqualified), the function eliminates the clc
    column with the smallest value from the sample data and tries again. This elimination is repeated
    until the number of remaining clc keys falls below config["clc_matchperc"] percent of the original
    clc keys. If still no match is found, an empty dict is returned.

    Finally, instead of returning the full matching row, the function returns a dictionary with
    the same structure as sampledata but with values taken from the matching row. For each clc key,
    both the VAL and the corresponding CD values are returned, and all numeric values are converted
    to plain Python numbers.

    Parameters:
      sampledata (dict): Expected to include keys such as 'area', 'slope', 'tri', etc.,
                         and for any key starting with "clc" whose value is a dict, those will be flattened.
                         Note: clc values are in pixels (each pixel = 10,000 m²).
      csvdata (pd.DataFrame): DataFrame with columns corresponding to the adapted sample data.
      config (dict): A configuration dictionary with at least the following keys:
          - "clc_minperc": The minimum percentage of the total area required for a clc value to be used.
          - "clc_matchperc": The minimum percentage (of the original number of clc keys) that must remain
                             during the iterative elimination.
          - "weighting_factors": A dict of weighting factors for columns, e.g.,
                {'area': 0, 'slope': 0.5, 'tri': 0.5, 'clc': 1}
          A weight of 0 means the column is ignored.

    Returns:
      dict: A dictionary with the same structure as sampledata but with values from the matching row,
            including both VAL and CD for clc keys (all as plain Python numbers), or an empty dict
            if no match is found.
    """

    def to_python(x):
        return x.item() if hasattr(x, "item") else x

    # 1. Flatten the sample data for keys starting with "clc"
    adapted_sampledata = {}
    for key, value in sampledata.items():
        if key.startswith("clc") and isinstance(value, dict):
            # Flatten with "_VAL_" inserted between key and subkey.
            for subkey, subvalue in value.items():
                adapted_sampledata[f"{key}_VAL_{subkey}"] = subvalue
        else:
            adapted_sampledata[key] = value

    # 2. Filter out clc values that are less than the configured percentage of the total area.
    # Each clc value is in pixels, where one pixel represents 10,000 m².
    # A clc value is kept if:
    #    (clc value * 10,000) >= (config["clc_minperc"]/100 * sampledata["area"])
    # Equivalently, clc value >= (sampledata["area"] * config["clc_minperc"]) / 1,000,000
    if "area" not in sampledata:
        raise ValueError("Missing 'area' in sampledata")
    area_value = sampledata["area"]
    clc_threshold = area_value * config["clc_minperc"] / 1000000.0

    # Remove clc keys with values below the threshold.
    keys_to_remove = [
        key
        for key in adapted_sampledata
        if key.startswith("clc") and adapted_sampledata[key] < clc_threshold
    ]
    for key in keys_to_remove:
        del adapted_sampledata[key]

    # Record original clc keys to enforce the elimination threshold later.
    original_clc_keys = [key for key in adapted_sampledata if key.startswith("clc")]
    initial_clc_count = len(original_clc_keys)

    # Define a helper function that, given the current adapted sample data,
    # computes normalization parameters, r_square values, and returns the best match.
    def attempt_match(adapted_data):
        # Identify keys to normalize: intersection of adapted_data keys and csvdata columns.
        keys_to_normalize = [k for k in adapted_data if k in csvdata.columns]
        # Compute normalization parameters (mean and std) for each key using csvdata.
        normalization_params = {}
        for col in keys_to_normalize:
            mean = csvdata[col].mean()
            std = csvdata[col].std()
            normalization_params[col] = (mean, std)
        # Normalize the sample values.
        sample_norm = {}
        for col in keys_to_normalize:
            mean, std = normalization_params[col]
            sample_norm[col] = 0 if std == 0 else (adapted_data[col] - mean) / std

        # Helper: compute weighted r_square for a row.
        def compute_r_square(row):
            # Disqualify rows that have zero in any required clc column.
            for col in keys_to_normalize:
                if col.startswith("clc") and row[col] == 0:
                    return float("inf")
            diff_sum = 0
            for col in keys_to_normalize:
                # Use the 'clc' weight for all clc columns; otherwise, use the column-specific weight.
                if col.startswith("clc"):
                    weight = config["weighting_factors"].get("clc", 1)
                else:
                    weight = config["weighting_factors"].get(col, 1)
                if weight == 0:
                    continue
                mean, std = normalization_params[col]
                row_norm = 0 if std == 0 else (row[col] - mean) / std
                diff = row_norm - sample_norm[col]
                diff_sum += weight * (diff**2)
            return diff_sum

        csvdata["r_square"] = csvdata.apply(lambda row: compute_r_square(row), axis=1)
        best_row = csvdata.loc[csvdata["r_square"].idxmin()]
        best_rsq = csvdata["r_square"].min()
        return best_row, best_rsq

    # 3. Try to find a match.
    best_match, best_rsq = attempt_match(adapted_sampledata)

    # 4. If no match is found (i.e. best_rsq is infinite), iteratively eliminate the smallest clc value.
    while best_rsq == float("inf"):
        remaining_clc_keys = [k for k in adapted_sampledata if k.startswith("clc")]
        if (
            len(remaining_clc_keys)
            < initial_clc_count * config["clc_matchperc"] / 100.0
        ):
            print("Not enough clc keys remain, no valid match found.")
            return {}
        smallest_key = min(remaining_clc_keys, key=lambda k: adapted_sampledata[k])
        print(
            f"Eliminating smallest clc key: {smallest_key} with value {adapted_sampledata[smallest_key]}"
        )
        del adapted_sampledata[smallest_key]
        best_match, best_rsq = attempt_match(adapted_sampledata)

    # Debug output: Display all rows with their corresponding r_square values.
    # Save the debug information to a CSV file.
    # debug_file_path = "debug.csv"
    # csvdata.to_csv(debug_file_path, index=False)

    # 5. Reconstruct the output dictionary with the same structure as sampledata,
    #    but with values taken from the best matching row.
    #    For each clc key, return both the VAL and the corresponding CD values.
    output = {}
    for key, value in sampledata.items():
        if key.startswith("clc") and isinstance(value, dict):
            new_subdict = {}
            for subkey in value.keys():
                flat_val_key = f"{key}_VAL_{subkey}"
                flat_cd_key = f"{key}_CD_{subkey}"
                new_val = (
                    best_match[flat_val_key]
                    if flat_val_key in best_match
                    else value[subkey]
                )
                new_cd = best_match[flat_cd_key] if flat_cd_key in best_match else None
                # Convert to plain Python numbers.
                new_subdict[subkey] = {
                    "VAL": to_python(new_val),
                    "CD": to_python(new_cd),
                }
            output[key] = new_subdict
        else:
            output[key] = to_python(best_match[key]) if key in best_match else value

    return output


def translate_match_results(sampledata, match_result, clclookup):
    """
    Translates clc codes and builds a new result dictionary from the input sampledata,
    the matching result from the previous function, and a clclookup DataFrame.

    The clclookup DataFrame is assumed to have columns including "Value" and "Name".
    The lookup is performed by matching the "Value" column with the clc code (the subkey).
    If a match is found, the corresponding "Name" is used.

    The returned dictionary has the following structure:
      {
        'area': <sampledata['area'] / 10000 (rounded to int)>,
        <other non-clc keys from sampledata (rounded to int)>,
        'clcsample': {<translated name>: <original clc value>, ...},
        'clcmatch': {<translated name>: <matching VAL value>, ...},
        'clcmatchchange': {<translated name>: <matching CD value>, ...},
        'clcsamplechange': {<translated name>: <(clcmatchchange/clcmatch)*clcsample>, ...},
        'clcsamplecperc': {<translated name>: <(clcsamplechange/clcsample)*100, rounded to 1 decimal>, ...}
      }

    All numeric values are converted to plain Python numbers and rounded to integers,
    except for clcsamplecperc which is kept as a float with one decimal.

    Parameters:
      sampledata (dict): The original sample data dictionary.
         Example:
           {
             'area': 12215944,
             'clc100': {16: 158, 24: 573, ...},
             'slope': 10,
             'tri': 31,
             ... (other keys)
           }
      match_result (dict): The result from the matching function, with clc keys structured as:
         {
           'area': ...,
           'clc100': {16: {"VAL": <match_val>, "CD": <change_val>}, ...},
           'slope': ...,
           'tri': ...,
           ... (other keys)
         }
      clclookup (pd.DataFrame): A DataFrame used for translating clc codes.
         Expected structure (example):
             Value  Code                             Name  Category  CatName
          0      1   111        Continuous urban fabric         1     Urban
          1      2   112     Discontinuous urban fabric         1       NaN
          2      3   121  Industrial or commercial units...   1       NaN

    Returns:
      dict: A dictionary with the described structure and with all numeric values (except clcsamplecperc)
            rounded to int. The clcsamplecperc values are floats with one digit after the decimal.
    """

    def to_python(x):
        """Convert numpy scalars to plain Python numbers."""
        return x.item() if hasattr(x, "item") else x

    def translate_clc(code):
        """
        Lookup the clc code in the clclookup DataFrame by matching the 'Value' column.
        If a row is found, return the corresponding 'Name' column value.
        Otherwise, return the code as a string.
        """
        code = to_python(code)
        lookup_row = clclookup[clclookup["Value"] == code]
        if not lookup_row.empty:
            return lookup_row.iloc[0]["Name"]
        return str(code)

    def round_all_numbers(obj):
        """Recursively round all numeric values in obj to int."""
        if isinstance(obj, dict):
            return {k: round_all_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_all_numbers(x) for x in obj]
        elif isinstance(obj, (int, float)):
            return int(round(obj))
        else:
            return obj

    output = {}

    # Set 'area' to be the sampledata area divided by 10000.
    if "area" in sampledata:
        output["area"] = to_python(sampledata["area"]) / 10000.0

    # Copy any non-clc keys (except 'area') from sampledata.
    for key, value in sampledata.items():
        if not key.startswith("clc") and key != "area":
            output[key] = to_python(value)

    # Build dictionaries for clc values.
    clcsample = {}
    clcmatch = {}
    clcmatchchange = {}

    # Process each clc key in sampledata.
    for key, clc_dict in sampledata.items():
        if key.startswith("clc") and isinstance(clc_dict, dict):
            for subkey, original_value in clc_dict.items():
                translated_name = translate_clc(subkey)
                clcsample[translated_name] = to_python(original_value)
                # For the matching result, we expect the same clc key structure:
                # match_result[key] is a dict with subkeys mapping to dicts containing "VAL" and "CD".
                if key in match_result and subkey in match_result[key]:
                    clcmatch[translated_name] = to_python(
                        match_result[key][subkey]["VAL"]
                    )
                    clcmatchchange[translated_name] = to_python(
                        match_result[key][subkey]["CD"]
                    )
                else:
                    clcmatch[translated_name] = None
                    clcmatchchange[translated_name] = None

    # Compute clcsamplechange as (clcmatchchange/clcmatch)*clcsample for each clc value.
    clcsamplechange = {}
    for name in clcsample:
        match_val = clcmatch.get(name)
        match_change = clcmatchchange.get(name)
        sample_val = clcsample.get(name)
        if match_val is None or match_val == 0:
            clcsamplechange[name] = None
        else:
            clcsamplechange[name] = (match_change / match_val) * sample_val

    output["clcsample"] = clcsample
    output["clcmatch"] = clcmatch
    output["clcmatchchange"] = clcmatchchange
    output["clcsamplechange"] = clcsamplechange

    # Round all numeric values (except we want to keep clcsamplecperc as float later).
    output = round_all_numbers(output)

    # Now compute clcsamplecperc as (clcsamplechange / clcsample)*100 for each clc value.
    # The result is kept as a float with one decimal.
    clcsamplecperc = {}
    for name, sample_val in output["clcsample"].items():
        if (
            sample_val is None
            or sample_val == 0
            or output["clcsamplechange"].get(name) is None
        ):
            clcsamplecperc[name] = None
        else:
            clcsamplecperc[name] = round(
                (output["clcsamplechange"][name] / sample_val) * 100, 1
            )

    output["clcsamplecperc"] = clcsamplecperc

    return output


def main(
    yaml_filename: str,
    data: dict,
    geojson_dict: dict,
    DEBUG=False,
):
    """
    Main function for the windlulc package that processes the YAML configuration
    and GeoJSON data.

    Args:
        yaml_filename (str): The path to the YAML configuration file.
        geojson_dict (dict): The GeoJSON dictionary (for example, turbines data).
    """
    # Read and display the YAML configuration
    config = read_yaml_config(PACKAGE_DIR / yaml_filename)
    if DEBUG:
        print("YAML Configuration:")
        print(config)

    clclookup_filename = os.path.join(PACKAGE_DIR, "data", "CLC.xlsx")
    clcdata_filename = os.path.join(
        PACKAGE_DIR, "data", config["hulltype"] + ".feather"
    )

    clclookup = pd.read_excel(clclookup_filename, sheet_name=config["clctype"])
    clcdata = pd.read_feather(clcdata_filename)

    # Convert the turbines GeoJSON to a GeoPandas GeoDataFrame and display it
    turbines_gdf = turbines_geojson_to_gdf(geojson_dict)
    turbines_gdf = turbines_gdf.to_crs(config["crs"])
    # calculate average distance between turbines
    turbines_gdf["buffer"] = cluster_distance(turbines_gdf["geometry"])
    # create a hull around the turbines (singlebuffer, concave or convex) also using the average distance
    hull_gdf = create_hulls(turbines_gdf, hull_type=config["hulltype"])
    # begin the result dictionary with prefilled area
    sampleresult = {"area": int(hull_gdf["geometry"][0].area)}
    # loop throug all raster and vectorfiles defined
    for name, values in data.items():
        if values["type"] == "clc" and name != config["clctype"]:
            continue
        if values["type"] in ("raster", "clc"):
            sampleres = sample_raster_values_within_polygon(
                values["path"], hull_gdf, values["result_type"]
            )
        sampleresult[name] = sampleres
    if DEBUG:
        print(sampleresult)

    matchresult = find_best_matching_row(sampleresult, clcdata, config)
    if DEBUG:
        print(matchresult)
    mapresult = translate_match_results(sampleresult, matchresult, clclookup)
    if DEBUG:
        print(mapresult)
    hull_dict = json.loads(hull_gdf.to_json())
    mapresult["hull"] = hull_dict
    return mapresult
