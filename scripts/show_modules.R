# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_fname = args[[1]]
input_option = args[[2]]

# Get gene names.
genes = read.delim("data/gene_annotations.tsv.gz")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)

cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-1]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

mod = as.matrix(read.table(input_fname))
rownames(mod) = names

if (input_option == "paired") {
   mod = mod[,c(1,3,5)] - mod[,c(2,4,6)]
}

for (i in 1:ncol(mod)) {
   cat("=======================\n")
   cat(paste("Module", i, "\n"))
   print(head(sort(mod[,i], decreasing=TRUE), 32))
   cat("---\n")
   print(head(sort(mod[,i], decreasing=FALSE), 32))
}
