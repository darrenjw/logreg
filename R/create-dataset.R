## create-dataset.R
## Script to save the Pima Indians training data in parquet format

if (!require("pacman")) install.packages("pacman")
pacman::p_load("MASS", "arrow")

write_parquet(Pima.tr, file.path("..", "pima.parquet"))


## eof

