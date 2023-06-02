# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_fname_1 = args[[1]]
input_fname_2 = args[[2]]
output_fname = args[[3]]

library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

plate_colors = c("#4a8337", "#6bac21", "#ddd48f", "#cda989", "#804012")
names(plate_colors) = c("P2449", "P2458", "P2769", "P2770", "P2771")

ref = read.delim(input_fname_1, row.names=1)

theta = as.matrix(read.table(input_fname_2))
vals = array(rgamma(n=1000*length(theta), shape=theta),
   dim=c(nrow(theta),2,1000))
sums = apply(X=vals, MARGIN=c(1,3), FUN=sum)

itvl = apply(X=vals[,1,]/sums, MARGIN=1, FUN=quantile, probs=c(.025,.975))

# Gather plate info.
plate = sub("_.*", "", rownames(ref))

pdf(output_fname, height=4, width=8 * nrow(theta) / 152)
showtext_begin()

par(mar=c(3.1,2.1,0.7,0))

plot(c(0,nrow(theta)), c(0,1), type="n",
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", panel.first=grid())

rect(xleft=0:(nrow(theta)-1), xright=1:nrow(theta),
   ybottom=itvl[1,], ytop=itvl[2,], col=plate_colors[plate], border=NA)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=diffinv(as.vector(table(plate))))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
   line=-1)
title(xlab="Cell number", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Module A", line=1, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()
