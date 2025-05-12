#!/usr/bin/env python
# Extracts the desired parameters from grib files, according to input lightning data.

import os
import sys
import socket
import pickle
import logging
import multiprocessing

import pandas
import pygrib


VERSION     = "3.0.0"
TAIL        = 24 * 3 # 3 days

# Margin is the box size around each lightning strike to get parameters from (in km).
MARGIN      = 3
MAX_MARGIN  = 6

# Expected variables in the grib files
# (parameter_id, parameter_name)
GRIB_FIELDS = {
    1:   "pressure",
    11:  "temperature",
    12:  "wet bulb temperature",
    15:  "maximum temperature",
    16:  "minimum temperature",
    20:  "visibility",
    32:  "wind gust",
    33:  "east wind",
    34:  "north wind",
    52:  "humidity",
    71:  "total cloud cover",
    73:  "low cloud cover",
    74:  "medium cloud cover",
    75:  "high cloud cover",
    77:  "fraction of significant clouds",
    #78:  "base of significant clouds above ground",
    78:  "base of significant clouds above sea",
    79:  "top of significant clouds",
    144: "frozen part of precipitation",
    145: "precipitation type",
    146: "precipitation sort",
    162: "12 hour precipitation",
    164: "24 hour precipitation",
    165: "precipitation",
    167: "3 hour precipitation",
    172: "12 hour snow",
    174: "24 hour snow",
    175: "snowfall",
    177: "3 hour snow",
}


# Parse arguments
if len(sys.argv) < 4:
    logging.critical("Received insufficient number of arguments")
    exit(1)

path_df_light = sys.argv[1]
output_dir    = sys.argv[2]
data_paths    = sys.argv[3:]

path_df_light_name = os.path.basename(path_df_light)
output_filepath = os.path.join(
    output_dir,
    path_df_light_name.replace("light", "mesan"),
)

output_bak = os.path.splitext(output_filepath)[0] + ".pkl"
output_log = os.path.splitext(output_filepath)[0] + ".log"


# Set up logging system
logging.basicConfig(
    level = logging.INFO,
    datefmt = "%Y-%m-%d %H:%M:%S",
    format = "[%(asctime)s] %(levelname)s: %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_log, mode="w", encoding="utf-8"),
    ]
)
hostname = socket.gethostname()
logging.info(f"Starting extraction for {path_df_light_name} on {hostname}")
logging.info(f"Script v{VERSION}")


def empty_sample():
    """Returns an empty observation"""
    return [None] * len(GRIB_FIELDS)


def get_sample(grbs, lat, lon):
    """Extracts a single observation"""
    filename = os.path.basename(grbs.name)

    sample = []
    for param_id, field in GRIB_FIELDS.items():

        try:
            if field.endswith("above ground"):
                results = grbs.select(indicatorOfParameter=param_id, typeOfLevel="heightAboveGround")
            elif field.endswith("above sea"):
                results = grbs.select(indicatorOfParameter=param_id, typeOfLevel="heightAboveSea")
            else:
                results = grbs.select(indicatorOfParameter=param_id)
        except ValueError:
            logging.warning(f"WARN File {filename}: is missing field '{field}'")
            sample.append(None)
            continue

        if len(results) == 0:
            logging.warning(f"File {filename}: is missing field '{field}'")
            sample.append(None)
            continue
        if len(results) > 1:
            logging.warning(f"File {filename}: field '{field}' got multiple matches")
        grb = results[0]

        for i in range(MARGIN, MAX_MARGIN+1):
            margin_deg = i * 0.008992806

            data, _, _ = grb.data(
                lat1 = lat - margin_deg,
                lat2 = lat + margin_deg,
                lon1 = lon - margin_deg,
                lon2 = lon + margin_deg,
            )

            if len(data) != 0:
                sample.append(data.mean())
                break

        else:
            logging.warning(f"File {filename}: field '{field}' lacks measurements within margin")
            sample.append(None)

    return sample


def find_file(name):
    for path in data_paths:
        filepath = os.path.join(path, name)
        if os.path.exists(filepath):
            return filepath
    return None


def get_samples(samples):
    """Extracts all required samples from file"""

    tim = samples[0][1]
    filename = tim.strftime("MESAN_%Y%m%d%H%M+000H00M")
    try:
        path = find_file(filename)
        if path is None:
            raise RuntimeError()
        grbs = pygrib.open(path)
    except Exception:
        logging.error(f"File {filename}: could be opened")
        return (idx, tim, *empty_sample())

    logging.info(f"Working on {filename}")
    results = []
    for idx, tim, lat, lon in samples:
        results.append((idx, tim, *get_sample(grbs, lat, lon)))

    grbs.close()
    return results


# Load lightning dataset
df_light = pandas.read_csv(path_df_light)
df_light["timestamp"] = pandas.to_datetime(df_light["timestamp"])


# Extract lightning strikes as tasks to process
tasks = {}
for _, row in df_light.iterrows():
    for hour in range(TAIL + 1):
        new_timestamp = row["timestamp"] - pandas.to_timedelta(hour, unit="h")
        task = (row["index"], new_timestamp, row["latitude"], row["longitude"])
        if new_timestamp in tasks:
            tasks[new_timestamp].append(task)
        else:
            tasks[new_timestamp] = [task]


# Process tasks
results = []
def mp_process(task): return get_samples(task)
with multiprocessing.Pool() as pool:
    results = pool.map(mp_process, tasks.values(), chunksize=8)

merged = []
for sublist in results:
    merged.extend(sublist)


# Save the resulting dataset
try:
    columns = ["light index", "timestamp"] + list(GRIB_FIELDS.values())
    dataset_mesan = pandas.DataFrame(merged, columns=columns)
    dataset_mesan = dataset_mesan.sort_values(by=["light index", "timestamp"])
    dataset_mesan.to_csv(output_filepath, index=False)
except Exception as e:
    logging.error(f"Could not convert array to dataframe ({e}), saving as pickle-object instead")
    with open(output_bak, "wb") as file:
        pickle.dump(results, file)
finally:
    logging.info(f"Finished extraction for {path_df_light_name} on {hostname}")


