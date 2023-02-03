# Input files (general LDA on three groups, no batch-effect correction).
out = as.matrix(read.table("out-alive-no-bec-no-HIV.txt", row.names=1))
wfreq = as.matrix(read.table("wfreq-alive-no-bec-no-HIV.txt"))

out = out[,c(1,3,2)]
wfreq = wfreq[c(1,3,2),] # SAHA, DMSO, PMA

# Get gene names (without version).
genes = read.delim("data/gene_annotations.tsv.gz")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-c(1,ncol(cells))]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

colnames(wfreq) = names

# Use weird formula...
score = function(w, idx) {
   w = w / rowSums(w) # Read probs for each signature.
   ranks = apply(w, rank, MARGIN=2)
   scores = (w[idx,] - w[ranks == 2])^2 / w[idx,] / (1-w[idx,])
   scores[ranks[idx,] != 3] = 0
   return(scores)
}

xS1 = score(wfreq, 1) # SAHA
xS2 = score(wfreq, 2) # DMSO
xS3 = score(wfreq, 3) # PMA

names(xS1) = names(xS2) = names(xS3) = names
SAHA = 100 * sort(xS1, decreasing=TRUE) / max(xS1)
DMSO = 100 * sort(xS2, decreasing=TRUE) / max(xS2)
PMA = 100 * sort(xS3, decreasing=TRUE) / max(xS3)

SAHA.pairs = apply(data.frame(as.integer(SAHA), names(SAHA)),
   MARGIN=1, FUN=paste, collapse=",")
DMSO.pairs = apply(data.frame(as.integer(DMSO), names(DMSO)),
   MARGIN=1, FUN=paste, collapse=",")
PMA.pairs = apply(data.frame(as.integer(PMA), names(PMA)),
   MARGIN=1, FUN=paste, collapse=",")
write(head(SAHA.pairs, 250), ncol=1, file="SAHA-weights.csv")
write(head(DMSO.pairs, 250), ncol=1, file="DMSO-weights.csv")
write(head(PMA.pairs, 250), ncol=1, file="PMA-weights.csv")

# Feed files to wordclouds.com
# https://www.wordclouds.com/

# Change names.
names(xS1) = names(xS2) = names(xS3) = colnames(cells)[-c(1,ncol(cells))]
SAHA = 100 * sort(xS1, decreasing=TRUE)
DMSO = 100 * sort(xS2, decreasing=TRUE)
PMA = 100 * sort(xS3, decreasing=TRUE)

write(head(gsub("\\.[0-9]*", "", names(SAHA)), 250), ncol=1,
   file="SAHA-gene_ids.txt")
write(head(gsub("\\.[0-9]*", "", names(DMSO)), 250), ncol=1,
   file="DMSO-gene_ids.txt")
write(head(gsub("\\.[0-9]*", "", names(PMA)), 250), ncol=1,
   file="PMA-gene_ids.txt")

# Feed files to DAVID.
#https://david.ncifcrf.gov/tools.jsp
