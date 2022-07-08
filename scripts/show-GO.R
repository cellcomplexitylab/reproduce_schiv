library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

args = commandArgs(trailingOnly=TRUE)

input_fname = args[[1]]
output_fname = args[[2]]

data = read.delim(input_fname)

# Keep every term with a corrected p-value below 0.1.
data = subset(data, Benjamini < .1)

# Keep only GO terms (just to be sure).
data = subset(data, grepl("^GO:", data$Term))

#enrichment = round(data$Fold.Enrichment)

pdf(output_fname, height=3.5, width=8)
showtext_begin()

par(mar=c(3.1,0.5,0.5,0.5))

# Manual bar plot (keep at most 10).
bars = rev(head(-log10(data$Benjamini), 10))
labels = rev(head(data$Term, 10))
labels = gsub("~", ": ", labels)
# Truncate terms.
labels = gsub("^(.{40}[^[:space:]]*)[[:space:]].*$", "\\1 ...", labels)

top = ceiling(max(bars))
plot(top*c(-1,1), c(0,10), 
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n", type="n")
rect(xleft=0, ybottom=.2 + 0:(length(bars)-1), xright=bars,
   ytop=-.2 + 1:length(bars), border=NA, col=c("gray70", "gray80"))
text(x=-.05 * top, y=.5 + 0:(length(bars)-1), adj=c(1,.5), labels,
   family="Avenir Medium", cex=.8, col="gray30")

abline(v=seq(from=0, to=top, length.out=6), col="white", lwd=2)
axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20",
   at=seq(from=0, to=top, length.out=6))
#title(xlab="Fold enrichment", line=2, col.lab="gray30", family="Avenir Medium")
mtext(text="Log10 corrected p-value", col="gray30", family="Avenir Medium", side=1,
   line=2, at=top / 2)


showtext_end()
dev.off()
