#!/usr/bin/env Rscript
# Preprocesses the lightning dataset.

#options(browser = "firefox")

suppressMessages(library(tidyverse))
suppressMessages(library(jsonlite))
suppressMessages(library(glue))
suppressMessages(library(plotly))
set.seed(0)


# Close lightning strikes will be clustered and counted as a single lightning strike.
# Achieved by rounding the lat/lon to `BIN_SIZE` number of decimals.
# - `2` will cluster lightning within +- 5.56km (recommended)
# - `1` will cluster lightning within +- 55.6km
# - `0` will cluster lightning within +- 555.6km
BIN_SIZE <- 1

FILE <- "data.csv"


# Load dataset
df_light <- read_csv(FILE)


# Perform processing
df <- df_light %>%
	mutate(timestamp = make_datetime(year, month, day, hours, minutes, seconds)) %>%
	mutate(timestamp = floor_date(timestamp, unit="hour")) %>%
	rename(latitude = lat, longitude = lon) %>%
	select(timestamp, latitude, longitude) %>%
	mutate(lightning = TRUE) %>%
	drop_na() %>%
	mutate(latitude = round(latitude, BIN_SIZE), longitude = round(longitude, BIN_SIZE)) %>%
	arrange(timestamp, latitude, longitude) %>%
	mutate(index = row_number()) %>%
	rename_with(make.names, names(.)) %>%
	select(timestamp, latitude, longitude) %>%
	count(timestamp, latitude, longitude)


# Select a subsample
s <- as_datetime("2020-01-01")
e <- as_datetime("2020-02-01")
df <- df %>%
	filter(between(timestamp, s, e)) %>%
	mutate(timestamp = paste(timestamp))


# Plot data
plot <- plot_ly(
	x = df$longitude,
	y = df$latitude,
	z = df$n,
	color = df$timestamp,
	opacity = 0.8,
)

plot

