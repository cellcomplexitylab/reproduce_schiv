# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_fname = args[[1]]
output_fname = args[[2]]
swap = args[[3]]

library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

plate_colors = c("#4a8337cc", "#6bac21cc", "#ddd48fcc", "#cda989cc", "#804012cc")
names(plate_colors) = c("P2449", "P2458", "P2769", "P2770", "P2771")

cells = read.delim("alivecells.tsv", row.names=1)
cells = cells[,-1] / rowSums(cells[,-1])
HIV = cells[,ncol(cells), drop=FALSE]

LDA = read.table(input_fname, row.names=1)
if (swap) LDA = LDA[,2:1]

m = merge(LDA, HIV, by="row.names")
topics = as.matrix(m[,2:3])

vals = array(rgamma(n=1000*nrow(topics), shape=topics),
   dim=c(nrow(topics),2,1000))
sums = apply(X=vals, MARGIN=c(1,3), FUN=sum)

itvl = apply(X=vals[,1,]/sums, MARGIN=1, FUN=quantile, probs=c(.025,.975))

# Gather plate info.
plate = sub("_.*", "", m$Row.names)

pdf(output_fname, height=4, width=4)
showtext_begin()

par(mar=c(2.8,2.8,0.7,0))

ymax = 2 * ceiling(max(100*m$HIVmini) / 2)
plot(c(0,1), c(0,ymax), type="n",
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", panel.first=grid())

set.seed(123)
noise = rnorm(nrow(m), sd=ymax/100)
ord = sample(nrow(m))
rect(xleft=itvl[1,ord], xright=itvl[2,ord],
   ybottom = 100*m$HIVmini[ord]+noise-ymax/200,
   ytop = 100*m$HIVmini[ord]+noise+ymax/200,
   col=plate_colors[plate[ord]], border=NA)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(ylab="HIV transcripts (%)", line=1.5, col.lab="gray30", family="Avenir Medium")
title(xlab="Module A", line=1.5, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()
