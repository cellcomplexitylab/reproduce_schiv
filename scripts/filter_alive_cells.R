cells = read.delim("alivecells.tsv")
SAHA_cells = subset(cells, grepl("SAHA", label))
write.table(SAHA_cells, file="SAHA_cells.tsv", sep="\t",
   quote=FALSE, row.names=TRUE)
