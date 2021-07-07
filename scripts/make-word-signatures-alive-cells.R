wfreq = as.matrix(read.table("wfreq-alive-no-bec-no-HIV.txt"))
wfreq = wfreq[c(1,3,2),] # SAHA, DMSO, PMA

# Get gene names (without version).
genes = read.delim("data/gene_annotations.tsv")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-c(1,ncol(cells))]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

colnames(wfreq) = names

SAHA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)
PMA = sort(wfreq[3,] - colMeans(wfreq[-3,]), decreasing=TRUE)

SAHA.pairs = apply(data.frame(as.integer(SAHA), names(SAHA)), MARGIN=1,
   paste, collapse=",")
PMA.pairs = apply(data.frame(as.integer(PMA), names(PMA)), MARGIN=1,
   paste, collapse=",")
write(head(SAHA.pairs, 50), ncol=1, file="SAHA-weights.csv")
write(head(PMA.pairs, 50), ncol=1, file="PMA-weights.csv")

# Feed files to wordclouds.com
# https://www.wordclouds.com/

# Change column names.
colnames(wfreq) = colnames(cells)[-c(1,ncol(cells))]

SAHA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)
PMA = sort(wfreq[3,] - colMeans(wfreq[-3,]), decreasing=TRUE)

write(head(gsub("\\.[0-9]*", "", names(SAHA)), 250), ncol=1,
   file="SAHA-gene_ids.txt")
write(head(gsub("\\.[0-9]*", "", names(PMA)), 250), ncol=1,
   file="PMA-gene_ids.txt")

# Feed files to DAVID.
#https://david.ncifcrf.gov/tools.jsp
