cells = read.delim("allcells.tsv")
expr = as.matrix(cells[,-1])

# Normalize by total expression.
expr = expr / rowSums(expr)
pca = prcomp(expr, scale=TRUE)

cells = subset(cells, pca$x[,1] < 10)

write.table(cells, file="alivecells.tsv", sep="\t",
   quote=FALSE, row.names=TRUE)
