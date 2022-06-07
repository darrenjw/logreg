#!/usr/bin/env Rscript
## create-dataset.R
## Script to save the Pima Indians training data in parquet format

if (!require("pacman")) install.packages("pacman")
pacman::p_load("MASS", "arrow")

write_parquet(Pima.tr, file.path("..", "pima.parquet"))

## Also save in a simple text format for primitive languages...
write.table(Pima.tr, file.path("..", "pima.data"),
            row.names=FALSE, col.names=FALSE, quote=FALSE)

## eof

