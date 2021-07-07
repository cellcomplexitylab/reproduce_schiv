library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

type_colors = c("#6db6ff", "#b5dafe", "#fe6db6",
       "#feb5da", "#9830b1", "#b68dff") 
plate_colors = c("#4a8337", "#6bac21", "#ddd48f", "#cda989", "#804012")

cells = read.delim("allcells.tsv")

plate = as.factor(gsub("_.*", "", rownames(cells)))
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

expr = as.matrix(cells[,-1])
# Get total read count.
total = rowSums(expr)
# Normalize by total expression.
expr = expr / rowSums(expr)
pca = prcomp(expr, scale=TRUE)


pdf("pca-all-cells.pdf")
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(pca$x, panel.first=grid(), col="gray20",
     bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
rect(xleft=10, xright=60, ybottom=-30, ytop=30, border=NA, col="#00000010")
points(pca$x, pch=19, cex=.8, col=type_colors[type])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="PC-1 (a.u.)", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="PC-2 (a.u.)", line=2, col.lab="gray30", family="Avenir Medium")

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     col=type_colors, pch=19, cex=.8,
     legend=c("J-LatA2 / DMSO", "Jurkat / DMSO", "J-LatA2 / PMA",
              "Jurkat / PMA", "J-LatA2 / SAHA", "Jurkat / SAHA"))

showtext_end()
dev.off()


total_colors = plate_colors[plate]
total_colors[pca$x[,1] > 10] = "black"

pdf("all-cells-total-reads.pdf")
showtext_begin()

par(mar=c(3.1,2.1,0.7,0))

barplot(total / 1e3, bty="n", ylab="", xlab="", xaxt="n", yaxt="n",
   space=0, border=NA, col=total_colors)

segments(x0=0, x1=480, y0=(1:4)*200, y1=(1:4)*200, lty=2, col="gray20") 
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
      at=96 * 0:5)
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20",
      line=-1)
title(xlab="Cell number", line=2, col.lab="gray30", family="Avenir Medium")
title(ylab="Total amount of reads (thousands)", line=1, col.lab="gray30",
      family="Avenir Medium")

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     col=plate_colors, pch=19, cex=.8,
     legend=paste("Plate", 1:5, sep=" "))

showtext_end()
dev.off()
