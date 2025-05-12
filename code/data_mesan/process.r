#!/usr/bin/env Rscript
# Compiles the meteorological dataset.

message("Initializing environment")
suppressMessages(library(tidyverse))
suppressMessages(library(glue))
set.seed(0)


data_directory <- "./parts"
output_file    <- "./data.csv"


message("Reading files from disk")
df_mesan <- tibble()
files <- list.files(data_directory, full.names=TRUE)
files <- files[endsWith(files, ".csv")]
for (file in files) {
	message(glue("- Loading file: {file}"))
	df_mesan <- bind_rows(df_mesan, read_csv(file, show_col_types=FALSE))
}


message("Processing meteorological data")
df_mesan <- df_mesan %>%
	rename(`base of significant clouds` = `base of significant clouds above ground`) %>%
	rename_with(make.names, names(.)) %>%
	arrange(light.index, timestamp) %>%
	group_by(light.index) %>%
	fill(.direction="downup") %>%
	filter(if_all(everything(), ~ !any(is.na(.)))) %>%
	ungroup() %>%
	mutate(frozen.part.of.precipitation = frozen.part.of.precipitation + 9) %>%
	mutate(precipitation.type = precipitation.type + 9) %>%
	select(
		light.index,
		timestamp,
		pressure,
		temperature,
		visibility,
		east.wind,
		north.wind,
		humidity,
		low.cloud.cover,
		medium.cloud.cover,
		high.cloud.cover,
		fraction.of.significant.clouds,
		base.of.significant.clouds,
		top.of.significant.clouds,
		frozen.part.of.precipitation,
		precipitation.type,
		precipitation.sort,
		precipitation,
		snowfall,
	)


message("Checking for NA values")
na_rows <- df_mesan %>%
	filter(if_any(everything(), is.na)) %>%
	count(light.index)
if (nrow(na_rows) != 0) {
	warning("The dataset contains NA values")
	print(na_rows, n=Inf, w=Inf)
	stop("Exiting script")
}


message("Saving final dataset")
print(df_mesan)
write_csv(df_mesan, output_file)

