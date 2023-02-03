out = as.matrix(read.table("out-PMA.txt", row.names=1))
wfreq = as.matrix(read.table("wfreq-PMA.txt"))
# HIV signature is already first.

# Get gene names.
genes = read.delim("data/gene_annotations.tsv.gz")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-1]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

colnames(wfreq) = names

# Find the percent reads attributed to the signature.
score = function(w, idx) {
   w = w / rowSums(w)
   ranks = apply(w, rank, MARGIN=2)
   scores = (w[idx,] - w[ranks == 2])^2 / w[idx,] / (1-w[idx,])
   scores[ranks[idx,] != 3] = 0
   return(scores)
}

xS1 = score(wfreq, 1)
PMA = 100 * sort(xS1, decreasing=TRUE) / max(xS1)
#total = out %*% wfreq
#S1 = outer(out[,1], wfreq[1,])
#xS1 = colMeans(S1 / total)
#
#names(xS1) = names
#PMA = 100 * sort(xS1, decreasing=TRUE)

PMA.pairs = apply(data.frame(as.integer(PMA), names(PMA)), MARGIN=1,
   paste, collapse=",")
write(head(PMA.pairs, 250), ncol=1, file="PMA-signature-weights.csv")

# Feed files to wordclouds.com
# https://www.wordclouds.com/

names(xS1) = colnames(cells)[-1]

PMA = 100 * sort(xS1, decreasing=TRUE)
write(head(gsub("\\.[0-9]*", "", names(PMA)), 250), ncol=1,
   file="PMA-signature-gene_ids.txt")

# Feed files to DAVID.
#https://david.ncifcrf.gov/tools.jsp
