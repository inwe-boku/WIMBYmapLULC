from typing import Union

import geopandas as gpd
import numpy as np
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
          dict: {raster_value: count} if result_type is "count"
          float: average raster value if result_type is "mean"

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

    if valid_data.size == 0:
        pixel_count = {}
        mean_value = None
    else:
        unique, counts = np.unique(valid_data, return_counts=True)
        pixel_count = dict(zip(unique, counts))
        npixel_count = {int(k): int(v) for k, v in pixel_count.items()}
        mean_value = int(np.mean(valid_data))

    if result_type == "count":
        return npixel_count
    elif result_type == "mean":
        return mean_value
    else:
        raise ValueError("Invalid result_type. Choose 'count' or 'mean'.")


def main(yaml_filename: str, rasterdata: dict, geojson_dict: dict, DEBUG=False):
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

    # Convert the turbines GeoJSON to a GeoPandas GeoDataFrame and display it
    turbines_gdf = turbines_geojson_to_gdf(geojson_dict)
    turbines_gdf = turbines_gdf.to_crs(config["crs"])
    # calculate average distance between turbines
    turbines_gdf["buffer"] = cluster_distance(turbines_gdf["geometry"])
    # create a hull around the turbines (singlebuffer, concave or convex) also using the average distance
    hull_gdf = create_hulls(turbines_gdf, hull_type=config["hulltype"])

    result = {}

    for name, values in rasterdata.items():
        sampleres = sample_raster_values_within_polygon(
            values["path"], hull_gdf, values["result_type"]
        )
        result[name] = sampleres
    if DEBUG:
        print(result)
