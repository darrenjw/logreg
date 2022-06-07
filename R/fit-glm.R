#!/usr/bin/env Rscript
## fit-glm.R
## Read data from disk and fit logistic regression model using "glm"

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow")

df = read_parquet(file.path("..", "pima.parquet"))
print(head(df))
fit = glm(type ~ ., data = df, family = "binomial")
print(fit)



## eof

