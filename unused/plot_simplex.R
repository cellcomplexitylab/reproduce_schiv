# Confidence interval for Speaman rho:
# https://stats.stackexchange.com/q/18887/10849
library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

treatment_colors = c("#fed769", "#6bc0b7")

topics = read.table("out_DMSO_PMA.txt", row.names=1)
#topics = topics[,c(2,1,3)]
topics = as.matrix(topics / rowSums(topics))

cells = read.delim("alivecells.tsv")
treatment = as.factor(gsub("^[^+]*\\+", "", cells$label))
treatment = treatment[match(rownames(topics), rownames(cells))]

# Compute coordinates in a simplex.
xcoord = .577 * (topics[,2] - topics[,3])
ycoord = topics[,1]

pdf("simplex_topics_DMSO_PMA.pdf", width=6, height=6/1.155)

par(mar=c(0,0,0,0))

plot(xcoord, ycoord, col="gray20", xlim=c(-.577, .577), ylim=c(0,1),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xcoord, ycoord, pch=19, cex=.8,
       col=treatment_colors[1 + (treatment == "PMA")])
polygon(x=c(-.577, .577, 0, -.577), y=c(0, 0, 1, 0), border="gray80")

legend(x="topright", inset=0.02,
     bg="white", box.col="gray50",
     col=treatment_colors, pch=19, cex=.8,
     legend=c("J-Lat A2 + DMSO", "J-Lat A2 + PMA"))

dev.off()

topics = read.table("out_DMSO_SAHA.txt", row.names=1)
#topics = topics[,c(2,1,3)]
topics = as.matrix(topics / rowSums(topics))

cells = read.delim("alivecells.tsv")
treatment = as.factor(gsub("^[^+]*\\+", "", cells$label))
treatment = treatment[match(rownames(topics), rownames(cells))]

# Compute coordinates in a simplex.
xcoord = .577 * (topics[,2] - topics[,3])
ycoord = topics[,1]

pdf("simplex_topics_DMSO_SAHA.pdf", width=6, height=6/1.155)

par(mar=c(0,0,0,0))

plot(xcoord, ycoord, col="gray20", xlim=c(-.577, .577), ylim=c(0,1),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xcoord, ycoord, pch=19, cex=.8,
       col=treatment_colors[1 + (treatment == "SAHA")])
polygon(x=c(-.577, .577, 0, -.577), y=c(0, 0, 1, 0), border="gray80")

legend(x="topright", inset=0.02,
     bg="white", box.col="gray50",
     col=treatment_colors, pch=19, cex=.8,
     legend=c("J-Lat A2 + DMSO", "J-Lat A2 + SAHA"))

dev.off()
