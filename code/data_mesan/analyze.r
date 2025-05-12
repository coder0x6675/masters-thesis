#/usr/bin/env Rscript
# Follows the steps done in the explorative MESAN analysis.

suppressMessages(library(ggcorrplot))
suppressMessages(library(tidyverse))
suppressMessages(library(glue))
set.seed(0)


# Drop NA columns
df <- read_csv("df_mesan.csv")
df <- df %>% dplyr::select(-light.index, -timestamp)
df <- df %>% dplyr::select(-where(~any(is.na(.x)) ), -light.index, -timestamp, -precipitation.type, -precipitation.sort)


# Test parameters for normality
shapiro.test(df$temperature[1:5000])


# Create covariance matrix
m  <- df %>% scale() %>% cor(method="spearman")
mp <- df %>% scale() %>% cor_pmat(method="spearman")


# Plot the covariance matrix
# Ordered using the hclust function
p <- ggcorrplot(round(m, 1), p.mat=round(mp, 1), hc.order=TRUE, type="lower", insig="blank", lab=TRUE)
ggsave("./plot.jpg")


# Perform the PC analysis
pca <- df %>% prcomp(scale=TRUE)
variances <- summary(pca)$importance[2,]
rotations_weighted <- abs(pca$rotation) %*% diag(variances)
param_weights <- rowSums(rotations_weighted)
normalized <- (param_weights / sum(param_weights)) * 100
sorted <- sort(normalized, decreasing=TRUE)

t <- tibble(names(sorted), sorted, c(0, diff(sorted)), cumsum(sorted))
print(sorted)
print(t)

