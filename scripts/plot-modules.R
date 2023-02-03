library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

# Get gene names.
genes = read.delim("data/gene_annotations.tsv.gz")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
cells = read.delim("alivecells.tsv", row.names=1)
gene_id = colnames(cells)[-1]
gene_id = gsub("\\.[0-9]*", "", gene_id)
names = genes$gene_symbol[match(gene_id, genes$gene_id)]
names[is.na(names)] = gene_id[is.na(names)]

# Read in the Dirichlet coefficients.
SAHA = as.matrix(read.table("wfreq-SAHA-2.txt"))
PMA = as.matrix(read.table("wfreq-PMA-2.txt"))

# Transform coefficients into distributions.
SAHA = SAHA / rowSums(SAHA)
PMA = PMA / rowSums(PMA)

xSAHA = log10(SAHA[2,]) - log10(SAHA[1,])
xPMA = log10(PMA[2,]) - log10(PMA[1,])

# Top 15.
SAHAidx = head(order(xSAHA, decreasing=TRUE), 15)
PMAidx = head(order(xPMA, decreasing=TRUE), 15)

print(as.character(names[sort(SAHAidx)]))
print(as.character(names[sort(PMAidx)]))

# The other top 20.
SAHAidx_ = head(order(xSAHA, decreasing=FALSE), 15)
PMAidx_ = head(order(xPMA, decreasing=FALSE), 15)

print(as.character(names[sort(SAHAidx_)]))
print(as.character(names[sort(PMAidx_)]))

pdf("modules-SAHA.pdf", height=4, width=8)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(xSAHA, panel.first=grid(), type="n", ylim=c(-4,4),
     bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xSAHA, col="#9830b150", type="h")
points(SAHAidx, xSAHA[SAHAidx], col="#9830b1", type="h")
points(SAHAidx, xSAHA[SAHAidx], col="#9830b1", pch=19, cex=.4)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Genes (arbitrary order)",
      line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="log-ratio A / B (base 10)",
      line=2, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()


pdf("modules-PMA.pdf", height=4, width=8)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(xPMA, panel.first=grid(), type="n", ylim=c(-4,4),
     bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xPMA, col="#fe6db650", type="h")
points(PMAidx, xPMA[PMAidx], col="#fe6db6", type="h")
points(PMAidx, xPMA[PMAidx], col="#fe6db6", pch=19, cex=.4)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Genes (arbitrary order)",
      line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="log-ratio A / B (base 10)",
      line=2, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()
