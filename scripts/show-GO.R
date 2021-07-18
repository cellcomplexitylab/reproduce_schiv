library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

SAHA.DAVID = read.delim("SAHA-DAVID.txt")
PMA.DAVID = read.delim("PMA-DAVID.txt")

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

write.table(SAHA.DAVID, file="SAHA-GO.txt", sep="\t",
   quote=FALSE, row.names=FALSE)
write.table(PMA.DAVID, file="PMA-GO.txt", sep="\t",
   quote=FALSE, row.names=FALSE)

SAHA_colors = c("#fed769", "#fed7699e")

pdf("barplot-GO-SAHA.pdf", height=3, width=8)
showtext_begin()

par(mar=c(3.1,0.5,0.5,0.5))

# Manual bar plot.
bars = rev(head(SAHA.DAVID$enrichment, 10))
labels = rev(head(SAHA.DAVID$Term, 10))
labels = gsub("~", ": ", labels)
# Truncate terms.
labels = gsub("^(.{40}[^[:space:]]*)[[:space:]].*$", "\\1 ...", labels)

plot(c(-30,35), c(0,10), 
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", type="n")
rect(xleft=0, ybottom=.2 + 0:9, xright=bars,
   ytop=-.2 + 1:10, border=NA, col=SAHA_colors)
text(x=-1, y=.5 + 0:9, adj=c(1,.5), labels,
   family="Avenir Medium", cex=.8, col="gray30")

abline(v=(1:7)*5, col="white", lwd=2)
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0:7)*5)
#title(xlab="Fold enrichment", line=2, col.lab="gray30", family="Avenir Medium")
mtext(text="Fold enrichment", col="gray30", family="Avenir Medium", side=1,
   line=2, at=17.5)


showtext_end()
dev.off()


PMA_colors = c("#008f97", "#008f979e")

pdf("barplot-GO-PMA.pdf", height=3, width=8)
showtext_begin()

par(mar=c(3.1,0.5,0.5,0.5))

# Manual bar plot.
bars = rev(head(PMA.DAVID$enrichment, 10))
labels = rev(head(PMA.DAVID$Term, 10))
labels = gsub("~", ": ", labels)
# Truncate terms.
labels = gsub("^(.{40}[^[:space:]]*)[[:space:]].*$", "\\1 ...", labels)

plot(c(-30,35), c(0,10), 
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", type="n")
rect(xleft=0, ybottom=.2 + 0:9, xright=bars,
   ytop=-.2 + 1:10, border=NA, col=PMA_colors)
text(x=-1, y=.5 + 0:9, adj=c(1,.5), labels,
   family="Avenir Medium", cex=.8, col="gray30")

abline(v=(1:7)*5, col="white", lwd=2)
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=c(0:7)*5)
#title(xlab="Fold enrichment", line=2, col.lab="gray30", family="Avenir Medium")
mtext(text="Fold enrichment", col="gray30", family="Avenir Medium", side=1,
   line=2, at=17.5)


showtext_end()
dev.off()

