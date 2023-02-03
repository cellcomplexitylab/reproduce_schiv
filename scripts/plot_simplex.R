# Read input data from command line.
args = commandArgs(trailingOnly=TRUE)
input_fname = args[[1]]
output_fname = args[[2]]

library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

topic_colors = c("#9830b1", "#6db6ff", "#feb5da")
type_colors = c("#6db6ff", "#b5dafe", "#fe6db6",
       "#feb5da", "#9830b1", "#b68dff") 

topics = read.table(input_fname)
topics = as.matrix(topics / rowSums(topics))

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

pdf(output_fname, width=6, height=6/1.155)

par(mar=c(0,0,0,0))

plot(xcoord, ycoord, col="gray20", xlim=c(-.577, .577), ylim=c(0,1),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(xcoord, ycoord, pch=19, cex=.8, col=type_colors[type])
polygon(x=c(-.577, .577, 0, -.577), y=c(0, 0, 1, 0), border="gray80")

legend(x="topright", inset=0.02,
     bg="white", box.col="gray50",
     col=type_colors, pch=19, cex=.8,
     legend=c("J-Lat A2 + DMSO", "Jurkat + DMSO", "J-Lat A2 + PMA",
              "Jurkat + PMA", "J-Lat A2 + SAHA", "Jurkat + SAHA"))

dev.off()
