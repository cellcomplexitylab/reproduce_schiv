library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

topic_colors = c("#e1b168", "#b96f20", "#1b343c")

topics = read.table("out-PMA.txt")
topics = topics[,c(2,3,1)] # Put HIV state on top.
topics = as.matrix(topics / rowSums(topics))

pdf("bargraph-topics-PMA.pdf", height=4, width=8 * 116 / 152)
showtext_begin()

par(mar=c(3.1,2.1,0.7,0))

barplot(t(topics), bty="n", ylab="", xlab="", xaxt="n", yaxt="n",
   space=0, border=NA, col=topic_colors)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0, 39, 78, 116))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
   line=-1)
title(xlab="Cell number", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Signature breakdown", line=1, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()


topics = read.table("out-PMA-no-bec.txt")
topics = topics[,c(3,2,1)] # Put HIV state on top.
topics = as.matrix(topics / rowSums(topics))

pdf("bargraph-topics-PMA-no-bec.pdf", height=4, width=8 * 116 / 152)
showtext_begin()

par(mar=c(3.1,2.1,0.7,0))

barplot(t(topics), bty="n", ylab="", xlab="", xaxt="n", yaxt="n",
   space=0, border=NA, col=topic_colors)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0, 39, 78, 116))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
   line=-1)
title(xlab="Cell number", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Signature breakdown", line=1, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()
