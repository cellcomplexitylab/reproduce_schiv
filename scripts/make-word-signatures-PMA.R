wfreq = as.matrix(read.table("wfreq-PMA.txt"))
wfreq[is.na(wfreq)] = 0

# Get gene names.
genes = read.delim("data/gene_annotations.tsv")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-1]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

colnames(wfreq) = names

# Signature 1 has highest HIV response (accounts for ~70% of
# the HIV transcripts).
PMA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)
PMA.pairs = apply(data.frame(as.integer(PMA), names(PMA)), MARGIN=1,
   paste, collapse=",")
write(head(PMA.pairs, 50), ncol=1, file="PMA-signature-1-weights.csv")


# Signature 2 accounts for ~ 30% of HIV transcripts.
PMA = sort(wfreq[2,] - colMeans(wfreq[-2,]), decreasing=TRUE)
PMA.pairs = apply(data.frame(as.integer(PMA), names(PMA)), MARGIN=1,
   paste, collapse=",")
write(head(PMA.pairs, 50), ncol=1, file="PMA-signature-2-weights.csv")

# Feed files to wordclouds.com
# https://www.wordclouds.com/

# Reload wfreq to use gene IDs.
wfreq = as.matrix(read.table("wfreq-PMA.txt"))
wfreq[is.na(wfreq)] = 0

colnames(wfreq) = colnames(cells)[-1]

# Write signature 1.
PMA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)
write(head(gsub("\\.[0-9]*", "", names(PMA)), 250), ncol=1,
   file="PMA-signature-1-gene_ids.txt")

# Write signature 2.
PMA = sort(wfreq[2,] - colMeans(wfreq[-2,]), decreasing=TRUE)
write(head(gsub("\\.[0-9]*", "", names(PMA)), 250), ncol=1,
   file="PMA-signature-2-gene_ids.txt")

# Feed files to DAVID.
#https://david.ncifcrf.gov/tools.jsp
