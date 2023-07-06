data.table = read.delim("data/exprMatrix.tsv", row.names=1)
initial_labels = read.delim("data/sampleSheet.tsv")
initial_labels$cell = gsub("-", ".", initial_labels$cell)
initial_labels = initial_labels[order(initial_labels$cell),]

cell_type = gsub("\\+.*", "", initial_labels$label)
batch = gsub("_.*", "", initial_labels$cell)
group = gsub("[^+]*\\+", "", initial_labels$label)

# Remove genes with fewer than 1 reads per cell on average.
data.table = subset(data.table, rowMeans(data.table) > 1)

# Remove the bottom 50% less variable genes.
v = apply(data.table, MARGIN=1, var)
data.table = subset(data.table, v > median(v))

# Sort rows and columns.
data.table = data.table[,order(colnames(data.table))]
data.table = data.table[order(rownames(data.table)),]

# Put cells in rows and add labels.
stopifnot(all(initial_labels$cell == colnames(data.table)))
final.table = data.frame(ID=initial_labels$cell,
      cell_type=cell_type, batch=batch, group=group, t(data.table))

write.table(final.table, file="allcells.tsv", sep="\t",
      quote=FALSE, row.names=FALSE)
