data.table = read.delim("data/exprMatrix.tsv", row.names=1)
labels = read.delim("data/sampleSheet.tsv")
labels$cell = gsub("-", ".", labels$cell)
labels = labels[order(labels$cell),]

# Remove genes with fewer than 1 reads per cell on average.
data.table = subset(data.table, rowMeans(data.table) > 1)

# Remove the bottom 50% less variable genes.
v = apply(data.table, MARGIN=1, var)
data.table = subset(data.table, v > median(v))

# Sort rows and columns.
data.table = data.table[,order(colnames(data.table))]
data.table = data.table[order(rownames(data.table)),]

# Put cells in rows and add labels.
stopifnot(all(labels$cell == colnames(data.table)))
final.table = data.frame(label=labels$label, t(data.table))

write.table(final.table, file="allcells.tsv", sep="\t",
      quote=FALSE, row.names=TRUE)
