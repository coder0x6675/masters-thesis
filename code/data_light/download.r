#!/usr/bin/env Rscript
# Downloads the lightning dataset.
# Example URL: https://opendata-download-lightning.smhi.se/api/version/1.0/year/2016/month/06/day/01/data.json

suppressMessages(library(tidyverse))
suppressMessages(library(glue))
set.seed(0)


MIN_DATE <- as_datetime("2015-06-01")
MAX_DATE <- as_datetime("2023-06-01") - days(1)
OUTPUT_DIR <- "./raw"


setwd(OUTPUT_DIR)


message("Calculating missing files")
owned_files <- list.files()
datetimes   <- seq(MIN_DATE, MAX_DATE, by="day")
data_files  <- tibble(
	name  = format(datetimes, "light_%Y%m%d.json"),
	url   = format(datetimes, "https://opendata-download-lightning.smhi.se/api/version/1.0/year/%Y/month/%m/day/%d/data.json"),
	owned = name %in% owned_files,
)


# Estimating download size
count <- sum(!data_files$owned)
size  <- round((count*367903) / 1073741824, 2)
message(glue("{count} files are missing."))
message(glue("Estimated download size: {size} gib"))
cat("\n")


message("Downloading lightning data")
data_files <- data_files %>%
	rowwise() %>%
	mutate(status = if_else(owned, 0, tryCatch(
		download.file(url, name, mode="wb"),
		error=\(e)1, warning=\(e)1
	))) %>%
	mutate(owned = !status)


# Notify failed downloads
failed_files <- data_files %>%
	filter(status != 0) %>%
	select(name, status)
if (nrow(failed_files) != 0) {
	warning("The following files could not be downloaded:")
	print(failed_files, n=Inf, w=Inf)
}
else {
	message("Lightning dataset processed successfully")
}

