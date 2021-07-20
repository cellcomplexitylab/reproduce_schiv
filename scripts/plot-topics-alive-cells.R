# Confidence interval for Speaman rho:
# https://stats.stackexchange.com/q/18887/10849
library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

topic_colors = c("#fed769", "#6bc0b7", "#008f97")
type_colors = c("#6db6ff", "#b5dafe", "#fe6db6",
       "#feb5da", "#9830b1", "#b68dff") 

topics = read.table("out-alive-no-bec-no-HIV.txt", row.names=1)
topics = topics[,c(1,3,2)] # SAHA, DMSO, PMA
topics = as.matrix(topics / rowSums(topics))

pdf("bargraph-alive-cells-topics-no-bec-no-HIV.pdf", height=4, width=8)
showtext_begin()

par(mar=c(3.1,2.1,0.7,0))

barplot(t(topics), bty="n", ylab="", xlab="", xaxt="n", yaxt="n",
   space=0, border=NA, col=topic_colors)

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0, 59, 121, 211, 303, 392))
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
   line=-1)
title(xlab="Cell number", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Signature breakdown", line=1, col.lab="gray30", family="Avenir Medium")

showtext_end()
dev.off()


cells = read.delim("alivecells.tsv")
celltype = as.factor(gsub("\\+.*", "", cells$label))
treatment = as.factor(gsub("^[^+]*\\+", "", cells$label))
type = c(0,2,4)[treatment] + c(1,2)[celltype]

# Key to the types.
# 1: J-LatA2+DMSO
# 2: Jurkat+DMSO
# 3: J-LatA2+PMA
# 4: Jurkat+PMA
# 5: J-LatA2+SAHA
# 6: Jurkat+SAHA


# Compute coordinates in a simplex.
xcoord = .577 * (topics[,2] - topics[,3])
ycoord = topics[,1]

pdf("simplex-topics-alive-cells-no-bec-no-HIV.pdf", width=6, height=6/1.155)

par(mar=c(0,0,0,0))

plot(xcoord, ycoord, col="gray20", xlim=c(-.577, .577), ylim=c(0,1),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xcoord, ycoord, pch=19, cex=.8, col=type_colors[type])
polygon(x=c(-.577, .577, 0, -.577), y=c(0, 0, 1, 0), border="gray80")

legend(x="topright", inset=0.02,
     bg="white", box.col="gray50",
     col=type_colors, pch=19, cex=.8,
     legend=c("J-LatA2 / DMSO", "Jurkat / DMSO", "J-LatA2 / PMA",
              "Jurkat / PMA", "J-LatA2 / SAHA", "Jurkat / SAHA"))

dev.off()


# Total amount of reads.
idxSAHA = cells$label == "J-LatA2+SAHA"
SAHA.signature = 100 * topics[idxSAHA,1]
nreads = rowSums(cells[idxSAHA,2:ncol(cells)])
HIV.response = 100 * cells[idxSAHA, ncol(cells)] / nreads


pdf("SAHA-signature-vs-HIV.pdf", height=5, width=5)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(SAHA.signature, HIV.response, panel.first=grid(), xlim=c(40,80),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(SAHA.signature, HIV.response, pch=19, cex=.8, col=topic_colors[1])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Strength of the SAHA signature (%)", line=2, col.lab="gray30",
      family="Avenir Medium")
title(ylab="HIV expression (% total reads)", line=2, col.lab="gray30",
      family="Avenir Medium")

rho = round(cor(SAHA.signature, HIV.response, method="spearman"), 2)
xx = cbind(SAHA.signature, HIV.response)
set.seed(123)
boot = replicate(1000,
   cor(xx[sample(nrow(xx), replace=TRUE),], method="spearman")[1,2])
lo = round(sort(boot)[25], 2)
hi = round(sort(boot)[975], 2)
rhotext = paste("Spearman rho: ", rho,
   "\n95% bootstrap CI: (", lo, ", ", hi, ")", sep="", collapse="")
height = graphics::strheight(rhotext, cex=.8)
width = graphics::strwidth(rhotext, cex=.8)
rect(xleft=40, ybottom=9-height, xright=40+width, ytop=9,
     border=NA, col="white")
text(x=40, y=9, rhotext,
   family="Avenir Medium", col="gray40", adj=c(0,1), cex=.8)

showtext_end()
dev.off()


idxPMA = cells$label == "J-LatA2+PMA"
PMA.signature = 100 * topics[idxPMA,3]
nreads = rowSums(cells[idxPMA,2:ncol(cells)])
HIV.response = 100 * cells[idxPMA, ncol(cells)] / nreads
cor(PMA.signature, HIV.response, method="spearman")

pdf("PMA-signature-vs-HIV.pdf", height=5, width=5)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(PMA.signature, HIV.response, panel.first=grid(), xlim=c(30,80),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(PMA.signature, HIV.response, pch=19, cex=.8, col=topic_colors[3])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Strength of the PMA signature (%)", line=2, col.lab="gray30",
      family="Avenir Medium")
title(ylab="HIV expression (% total reads)", line=2, col.lab="gray30",
      family="Avenir Medium")

rho = round(cor(PMA.signature, HIV.response, method="spearman"), 2)
xx = cbind(PMA.signature, HIV.response)
set.seed(123)
boot = replicate(1000,
   cor(xx[sample(nrow(xx), replace=TRUE),], method="spearman")[1,2])
lo = round(sort(boot)[25], 2)
hi = round(sort(boot)[975], 2)
rhotext = paste("Spearman rho: ", rho,
   "\n95% bootstrap CI: (", lo, ", ", hi, ")", sep="", collapse="")
height = graphics::strheight(rhotext, cex=.8)
width = graphics::strwidth(rhotext, cex=.8)
rect(xleft=30, ybottom=3.5-height, xright=30+width, ytop=3.5,
     border=NA, col="white")
text(x=30, y=3.5, rhotext,
   family="Avenir Medium", col="gray40", adj=c(0,1), cex=.8)

showtext_end()
dev.off()
