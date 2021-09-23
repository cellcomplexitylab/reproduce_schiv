library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

SAHA = read.table("scramble-traces-SAHA.txt")
PMA = read.table("scramble-traces-PMA.txt")

yS = SAHA$V5
yP = PMA$V5

pdf("scramble-traces.pdf", width=10, height=5)
showtext_begin()

par(mar=c(3.1,3.1,0,0))
par(mfrow=c(1,2))

plot(rep(1:50, times=110), yS, ylim=c(2.3,3), type="n",
   panel.first=grid(), bty="n", xlab="", ylab="", xaxt="n", yaxt="n")
for (i in 11:110) {
   lines(yS[(1+(i-1)*50):(i*50)], col="#00000020", lwd=.5)
}
for (i in 1:10) {
   lines(yS[(1+(i-1)*50):(i*50)], col=2, lwd=.5)
}
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Iteration", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Loss (a.u.)", line=2, col.lab="gray30", family="Avenir Medium")

legend(x="topright", inset=.01, bg="white", box.col=NA,
     col=1:2, lwd=1, cex=.8, legend=c("scrambled", "original"))

plot(rep(1:50, times=110), yP, ylim=c(1.7,2.2), type="n",
   panel.first=grid(), bty="n", xlab="", ylab="", xaxt="n", yaxt="n")
for (i in 11:110) {
   lines(yP[(1+(i-1)*50):(i*50)], col="#00000020", lwd=.5)
}
for (i in 1:10) {
   lines(yP[(1+(i-1)*50):(i*50)], col=2, lwd=.5)
}
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Iteration", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Loss (a.u.)", line=2, col.lab="gray30", family="Avenir Medium")

legend(x="topright", inset=.01, bg="white", box.col=NA,
     col=1:2, lwd=1, cex=.8, legend=c("scrambled", "original"))
showtext_end()
dev.off()
