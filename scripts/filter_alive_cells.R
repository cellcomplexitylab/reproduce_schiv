# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
grepit = args[[1]]
output_fname = args[[2]]

cells = subset(read.delim("alivecells.tsv"),
   grepl(grepit, group))
write.table(cells, file=output_fname, sep="\t",
   quote=FALSE, row.names=FALSE)
