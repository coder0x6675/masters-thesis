#!/usr/bin/env Rscript
# Preprocesses the lightning dataset.

suppressMessages(library(tidyverse))
suppressMessages(library(jsonlite))
suppressMessages(library(glue))
set.seed(0)


INPUT_PATH  <- "./raw"
OUTPUT_DIR  <- "./parts"
OUTPUT_NAME <- "data"

# Earliest date for lightning data: 2014-03-26
# Earliest date for meteorological data: 2015-01-01
MIN_DATE <- as_datetime("2015-06-01")
MAX_DATE <- as_datetime("2023-06-01") - days(1)

# Close lightning strikes will be clustered and counted as a single lightning strike.
# Achieved by rounding the lat/lon to `BIN_SIZE` number of decimals.
# - `2` will cluster lightning within +- 5.56km (recommended)
# - `1` will cluster lightning within +- 55.6km
# - `0` will cluster lightning within +- 555.6km
BIN_SIZE <- 0

# Coordinates for city: Linköping
POINT_LAT <- 58.408730134762540
POINT_LON <- 15.619342170374372
BOX_SIZE  <- 300 #km. Set to `Inf` to use full dataset.

# Maximum area covered by the MESAN dataset
# a = 11_679_839 km²
MIN_LAT <- 52.30
MAX_LAT <- 71.50
MIN_LON <- -9.50
MAX_LON <- 39.70

km_to_deg <- function(x) x * 0.00899321
deg_to_km <- function(x) x / 0.00899321

box_size <- km_to_deg(BOX_SIZE)
min_lat  <- max(MIN_LAT, POINT_LAT - box_size)
min_lon  <- max(MIN_LON, POINT_LON - box_size)
max_lat  <- min(MAX_LAT, POINT_LAT + box_size)
max_lon  <- min(MAX_LON, POINT_LON + box_size)


setwd(OUTPUT_DIR)


path_df_light <- glue("{OUTPUT_NAME}.csv")
path_df_light_parts <- glue("{OUTPUT_NAME}_parts")
get_path_df_light_part <- function(i) {
	filename <- glue("{OUTPUT_NAME}_part_", str_pad(i, 4, pad="0"), ".csv")
	return(file.path(path_df_light_parts, filename))
}

dir.create(path_df_light_parts, showWarnings=FALSE)


message("Loading data from disk")
df_light <- tibble()
for (file in list.files(INPUT_PATH, full.names=TRUE)) {
	message(glue("- Loading file: {file}"))
	data <- read_json(file)
	if (length(data$values) == 0) {next}
	df_light <- bind_rows(df_light, as_tibble(data) %>% unnest_wider(values))
}


message("Processing positive observations")
df_light_positive <- df_light %>%
	filter(between(lat, min_lat, max_lat) & between(lon, min_lon, max_lon)) %>%
	mutate(timestamp = make_datetime(year, month, day, hours, minutes, seconds)) %>%
	mutate(timestamp = floor_date(timestamp, unit="hour")) %>%
	rename(latitude = lat, longitude = lon) %>%
	select(timestamp, latitude, longitude) %>%
	mutate(lightning = TRUE) %>%
	drop_na()


message("Generating negative observations")
n <- 2 * nrow(df_light_positive)
df_light_negative <- tibble(
	timestamp = floor_date(as_datetime(runif(n, MIN_DATE, MAX_DATE)), unit="hour"),
	latitude  = round(runif(n, min_lat, max_lat), digits=4),
	longitude = round(runif(n, min_lon, max_lon), digits=4),
	lightning = FALSE,
)


message("Merging positive and negative observations")
df_light <- bind_rows(df_light_positive, df_light_negative) %>%
	mutate(lat_aprox = round(latitude, BIN_SIZE), lon_aprox = round(longitude, BIN_SIZE)) %>%
	distinct(timestamp, lat_aprox, lon_aprox, .keep_all=TRUE)


message("Processing full dataset")
s <- sum(df_light$lightning)
df_light <- df_light %>%
	slice_head(n = s, by=lightning) %>%
	arrange(timestamp, latitude, longitude) %>%
	mutate(index = row_number()) %>%
	rename_with(make.names, names(.)) %>%
	select(index, timestamp, latitude, longitude, lightning)

stopifnot(all(count(df_light, lightning)$n == s))
stopifnot(!any(is.na(df_light)))


message("Saving final dataset")
print(df_light)
write_csv(df_light, path_df_light)


message("Saving dataset partitions")
partition_size <- 200
df_light %>%
	mutate(partition = (row_number()-1) %/% partition_size) %>%
	group_by(partition) %>%
	group_walk(\(df, i) write_csv(df, get_path_df_light_part(i[[1]])), .keep=FALSE)


message("Lightning dataset acquired successfully")

