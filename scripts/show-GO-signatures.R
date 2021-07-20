library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

SAHA.DAVID = read.delim("SAHA-signature-DAVID.txt")
PMA.DAVID = read.delim("PMA-signature-DAVID.txt")

# Sort by fold enrichment.
SAHA.DAVID = SAHA.DAVID[order(SAHA.DAVID$Fold.Enrichment, decreasing=TRUE),]
PMA.DAVID = PMA.DAVID[order(PMA.DAVID$Fold.Enrichment, decreasing=TRUE),]

# Keep every term with a p-value below 0.01 after correction.
SAHA.DAVID = subset(SAHA.DAVID, Benjamini < .01)
PMA.DAVID = subset(PMA.DAVID, Benjamini < .01)

# Keep only GO terms.
SAHA.DAVID = subset(SAHA.DAVID, grepl("^GO:", SAHA.DAVID$Term))
PMA.DAVID = subset(PMA.DAVID, grepl("^GO:", PMA.DAVID$Term))

SAHA.DAVID = data.frame(enrichment=round(SAHA.DAVID$Fold.Enrichment, 1),
   Term=SAHA.DAVID$Term)
PMA.DAVID = data.frame(enrichment=round(PMA.DAVID$Fold.Enrichment, 1),
   Term=PMA.DAVID$Term)

write.table(SAHA.DAVID, file="SAHA-signature-GO.txt", sep="\t",
   quote=FALSE, row.names=FALSE)
write.table(PMA.DAVID, file="PMA-signature-GO.txt", sep="\t",
   quote=FALSE, row.names=FALSE)

SAHA_colors = c("#e6d2eb", "#e6d2eb9e")

pdf("barplot-GO-SAHA-signature.pdf", height=3, width=8)
showtext_begin()

par(mar=c(3.1,0.5,0.5,0.5))

# Manual bar plot.
bars = rev(head(SAHA.DAVID$enrichment, 10))
labels = rev(head(SAHA.DAVID$Term, 10))
labels = gsub("~", ": ", labels)
labels = gsub("^(.{45}[^[:space:]]*)[[:space:]].*$", "\\1 ...", labels)

plot(c(-25,25), c(0,10), 
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", type="n")
rect(xleft=0, ybottom=.2 + 0:9, xright=bars,
   ytop=-.2 + 1:10, border=NA, col=SAHA_colors)
text(x=-1, y=.5 + 0:9, adj=c(1,.5), labels,
   family="Avenir Medium", cex=.8, col="gray30")

abline(v=(1:4)*5, col="white", lwd=2)
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0:5)*5)
mtext(text="Fold enrichment", col="gray30", family="Avenir Medium", side=1,
   line=2, at=12.5)


showtext_end()
dev.off()




PMA_colors = c("#fceaf0", "#fcdff0")

pdf("barplot-GO-PMA-signature.pdf", height=1, width=8)
showtext_begin()

par(mar=c(3.1,0.5,0.5,0.5))

# Manual bar plot.
bars = rev(head(PMA.DAVID$enrichment, 1))
labels = rev(head(PMA.DAVID$Term, 1))
labels = gsub("~", ": ", labels)
labels = gsub("^(.{45}[^[:space:]]*)[[:space:]].*$", "\\1 ...", labels)

plot(c(-25,25), c(0,1), 
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", type="n")
rect(xleft=0, ybottom=.2 + 0, xright=bars,
   ytop=-.2 + 1, border=NA, col=PMA_colors)
text(x=-1, y=.5 + 0, adj=c(1,.5), labels,
   family="Avenir Medium", cex=.8, col="gray30")

abline(v=(1:4)*5, col="white", lwd=2)
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0:5)*5)
mtext(text="Fold enrichment", col="gray30", family="Avenir Medium", side=1,
   line=2, at=12.5)

showtext_end()
dev.off()
