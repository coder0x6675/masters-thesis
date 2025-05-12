#!/usr/bin/env Rscript
# Prepares the LIGHT and MESAN datasets for model usage.

suppressMessages(library(tidyverse))
suppressMessages(library(glue))
set.seed(0)


FOLD_COUNT <- 10

get_part_name <- function(i) {
	glue("df_light/df_light_part_", str_pad(i, 2, pad="0"), ".csv")
}

path_df_light <- args[1]
path_df_mesan <- args[2]


message("Loading datasets from disk")
df_light <- read_csv(path_df_light)
df_mesan <- read_csv(path_df_mesan)


message("Deleting LIGHT instances not in MESAN")
df_light <- df_light %>%
	filter(index %in% df_mesan$light.index)


message("Rebalancing LIGHT dataset")
s <- min(table(df_light$lightning))
df_light <- df_light %>%
	slice_head(n = s, by=lightning)


message("Deleting MESAN instances not in LIGHT")
df_mesan <- df_mesan %>%
	filter(light.index %in% df_light$index)


message("Splitting LIGHT dataset")
df_light_pos <- df_light %>%
	filter(lightning == TRUE) %>%
	slice_sample(prop = 1)
df_light_neg <- df_light %>%
	filter(lightning == FALSE) %>%
	slice_sample(prop = 1)


message("Saving LIGHT dataset")
dir.create("df_light", showWarnings=FALSE)
n <- nrow(df_light) %/% (2*FOLD_COUNT)
for (i in 0:(FOLD_COUNT - 1)) {
	message(glue("Saving LIGHT part {i}"))
	s <- i * n ; e <- i*n + n - 1
	bind_rows(slice(df_light_pos, s:e), slice(df_light_neg, s:e)) %>%
		arrange(index, timestamp) %>%
		write_csv(get_part_name(i))
}


message("Saving MESAN dataset")
write_csv(df_mesan, "df_mesan.csv")

