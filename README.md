# WIMBYmapLULC
Plugin for the WIMBYmap

## Workflow

- inputs are taken from the wimbymap as geojson points
- the average distance between the turbine points are calculated
- based on the config a hull around the points is created (default = concave)
- within the hull clc values and other optional values are sampled (slope, ruggedness)
- based on the sampled values a matching row from WP1 (T1.2) results is returned with CD values
- a resulting dictionary is returned to the wimbymap

## Return dictionary

### Example

{'area': 724, 'slope': 1, 'tri': 3, 'clcsample': {'Non-irrigated arable land': 681, 'Pastures, meadows and other permanent grasslands under agricultural use': 30, 'Broad-leaved forest': 11}, 'clcmatch': {'Non-irrigated arable land': 790, 'Pastures, meadows and other permanent grasslands under agricultural use': 0, 'Broad-leaved forest': 0}, 'clcmatchchange': {'Non-irrigated arable land': 0, 'Pastures, meadows and other permanent grasslands under agricultural use': 0, 'Broad-leaved forest': 0}, 'clcsamplechange': {'Non-irrigated arable land': 0, 'Pastures, meadows and other permanent grasslands under agricultural use': None, 'Broad-leaved forest': None}, 'clcsamplecperc': {'Non-irrigated arable land': 0.0, 'Pastures, meadows and other permanent grasslands under agricultural use': None, 'Broad-leaved forest': None}, 'hull': {'type': 'FeatureCollection', 'features': [{'id': '0', 'type': 'Feature', 'properties': {'index': 0, 'buffer': 430, 'hull_id': 1}, 'geometry': {'type': 'Polygon', 'coordinates': [...], ]]}}], 'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3035'}}}}

- area in hectare
- slope in ° (optional)
- tri (ruggedness) (optional)
- clcsample subdict: the sampled values within the polygon in hectares
- clcmatch subdict: the matched values in hectares from the CD database
- clcmatchchange: the match CD values in hectares from the CD database
- clcsamplechange: the estimated CD values in hectares for the sample polygon
- clcsamplecperc: the estimated CD % change within the polygon for each category
- the hull geojson dictionary
