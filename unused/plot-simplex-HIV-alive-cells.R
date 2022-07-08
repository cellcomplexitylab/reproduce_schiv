topics = read.table("out-alive-no-bec-no-HIV.txt", row.names=1)
topics = topics[,c(1,3,2)] # SAHA, DMSO, PMA
topics = as.matrix(topics / rowSums(topics))

cells = read.delim("alivecells.tsv")
celltype = as.factor(gsub("\\+.*", "", cells$label))
treatment = as.factor(gsub("^[^+]*\\+", "", cells$label))
type = c(0,2,4)[treatment] + c(1,2)[celltype]


# Compute coordinates in a simplex.
xcoord = .577 * (topics[,2] - topics[,3])
ycoord = topics[,1]

pdf("simplex-topics-HIV.pdf", width=6, height=6/1.155)

par(mar=c(0,0,0,0))

plot(xcoord, ycoord, col="gray20", xlim=c(-.577, .577), ylim=c(0,1),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xcoord, ycoord, pch=19, cex=.8, col=(1+(cells$HIVmini > 1)))
polygon(x=c(-.577, .577, 0, -.577), y=c(0, 0, 1, 0), border="gray80")

legend(x="topright", inset=0.02,
     bg="white", box.col="gray50",
     col=1:2, pch=19, cex=.8,
     legend=c("HIV-", "HIV+"))

dev.off()
