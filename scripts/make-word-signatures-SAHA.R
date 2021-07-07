wfreq = as.matrix(read.table("wfreq-SAHA.txt"))
wfreq[is.na(wfreq)] = 0
wfreq = wfreq[c(3,1,2),]

# Get gene names.
genes = read.delim("data/gene_annotations.tsv")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-1]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

colnames(wfreq) = names

SAHA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)

SAHA.pairs = apply(data.frame(as.integer(SAHA), names(SAHA)), MARGIN=1,
   paste, collapse=",")
write(head(SAHA.pairs, 50), ncol=1, file="SAHA-signature-weights.csv")

# Feed files to wordclouds.com
# https://www.wordclouds.com/

# Reload wfreq to use gene IDs.
wfreq = as.matrix(read.table("wfreq-SAHA.txt"))
wfreq[is.na(wfreq)] = 0
wfreq = wfreq[c(3,1,2),]

colnames(wfreq) = colnames(cells)[-1]

SAHA = sort(wfreq[1,] - colMeans(wfreq[-1,]), decreasing=TRUE)

write(head(gsub("\\.[0-9]*", "", names(SAHA)), 250), ncol=1,
   file="SAHA-signature-gene_ids.txt")

# Feed files to DAVID.
#https://david.ncifcrf.gov/tools.jsp
