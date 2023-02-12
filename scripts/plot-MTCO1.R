# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_fname = args[[1]]
output_fname = args[[2]]

library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

plate_colors = c("#4a833744", "#6bac2144", "#ddd48f44", "#cda98944", "#80401244")
names(plate_colors) = c("P2449", "P2458", "P2769", "P2770", "P2771")

cells = read.delim("alivecells.tsv", row.names=1)
plate = sub("_.*", "", rownames(cells))
X = as.matrix(cells)[,-1]
MTCO1 = X[,4634]

smpl = as.matrix(read.table(input_fname))

pdf(output_fname)
showtext_begin()

par(mar=c(2.8,2.8,0.7,0))

plot(MTCO1, 1:length(MTCO1), type="n",
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", panel.first=grid())
for (i in 1:nrow(smpl)) {
   points(smpl[i,], i+rnorm(sd=.05, n=1000), pch=".",
   col=plate_colors[plate[i]])
}
points(MTCO1, 1:length(MTCO1), pch=19, cex=.5, col="gray15")

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
   at=diffinv(as.vector(table(plate))))
title(ylab="Cell", line=1.5, col.lab="gray30", family="Avenir Medium")
title(xlab="MT-CO1 expression (number of reads)", line=1.5, col.lab="gray30",
   family="Avenir Medium")

showtext_end()
dev.off()
