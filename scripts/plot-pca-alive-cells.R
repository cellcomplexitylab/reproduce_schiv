library(showtext)
font_add(family="Avenir Medium", regular="Avenir-Medium.ttf")

type_colors = c("#6db6ff", "#b5dafe", "#fe6db6",
       "#feb5da", "#9830b1", "#b68dff") 
plate_colors = c("#4a8337", "#6bac21", "#ddd48f", "#cda989", "#804012")

cells = read.delim("alivecells.tsv")

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

# Fraction explained variance.
fvar = round(100 * pca$sdev^2 / sum(pca$sdev^2))


pdf("pca-alive-cells.pdf", height=4, width=8)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(pca$x, panel.first=grid(), col="gray20",
     bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(pca$x, pch=19, cex=.8, col=type_colors[type])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab=paste("PC-1 (a.u.), ", fvar[1], "% variance", sep="", collapse=""),
      line=2, col.lab="gray30", family="Avenir Medium")
title(ylab=paste("PC-2 (a.u.), ", fvar[2], "% variance", sep="", collapse=""),
      line=2, col.lab="gray30", family="Avenir Medium")

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     # Change order to match the rest of the manuscript.
     col=type_colors[c(1,2,5,6,3,3)], pch=19, cex=.8,
     legend=c("J-Lat A2 + DMSO", "Jurkat + DMSO",
              "J-Lat A2 + SAHA", "Jurkat + SAHA",
              "J-Lat A2 + PMA",  "Jurkat + PMA")
)

showtext_end()
dev.off()

# Get the centroids.
DMSO = pca$x[type %in% c(1,2),1:2]
SAHA = pca$x[type %in% c(5,6),1:2]
PMA = pca$x[type %in% c(3,4),1:2]

DMSO_x = colMeans(DMSO)
SAHA_x = colMeans(SAHA)
PMA_x = colMeans(PMA)

# Get the effects (vectors).
SAHA_v = (SAHA_x - DMSO_x)
PMA_v = (PMA_x - DMSO_x)

SAHA_v = SAHA_v / (sqrt(sum(SAHA_v^2)))
PMA_v = PMA_v / (sqrt(sum(PMA_v^2)))

# Get the gene loadings.
SAHA_r = pca$rotation[,1:2] %*% SAHA_v
PMA_r = pca$rotation[,1:2] %*% PMA_v

top_SAHA = tail(sort(SAHA_r[,1]), 50)
top_SAHA_names = gsub("\\.[0-9]*", "", names(top_SAHA))
top_PMA = tail(sort(PMA_r[,1]), 50)
top_PMA_names = gsub("\\.[0-9]*", "", names(top_PMA))

write(top_SAHA_names, ncol=1, file="SAHA_up_gene_ids.txt")
#write(bot_SAHA, ncol=1, file="SAHA_dn_gene_ids.txt")
write(top_PMA_names, ncol=1, file="PMA_up_gene_ids.txt")
#write(bot_PMA, ncol=1, file="PMA_dn_gene_ids.txt")

# Write the weight files for https://www.wordclouds.com/
genes = read.delim("data/gene_annotations.tsv")
genes$gene_id = gsub("\\.[0-9]*", "", genes$gene_id)
names_SAHA = genes$gene_symbol[match(top_SAHA_names, genes$gene_id)]
names_PMA = genes$gene_symbol[match(top_PMA_names, genes$gene_id)]

S = data.frame(score=round(100*top_SAHA/max(top_SAHA)), name=names_SAHA)
write.table(S, sep=",", quote=FALSE, row.names=FALSE, col.names=FALSE,
   file="SAHA_weights.csv")
P = data.frame(score=round(100*top_PMA/max(top_PMA)), name=names_PMA)
write.table(P, sep=",", quote=FALSE, row.names=FALSE, col.names=FALSE,
   file="PMA_weights.csv")


SAHA.response = SAHA.response = t(t(SAHA) - DMSO_x)
SAHA.response = SAHA.response %*% SAHA_v

HIV.response = expr[type %in% c(5,6),ncol(expr)]

pdf("SAHA-signature-vs-HIV.pdf", height=5, width=5)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(SAHA.response, 100*HIV.response, panel.first=grid(),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(SAHA.response, 100*HIV.response, pch=19, cex=.8,
   col=type_colors[type[type %in% c(5,6)]])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Strength of the SAHA signature (a.u.)", line=2, col.lab="gray30",
      family="Avenir Medium")
title(ylab="Reads in GFP-nef (%)", line=2, col.lab="gray30",
      family="Avenir Medium")

# Compute correlation with J-Lat A2 only (Jurkat have
# no HIV, so they artifically lower the correlation).
stype = type[type %in% c(5,6)]
SAHA.response = SAHA.response[stype == 5]
HIV.response = HIV.response[stype == 5]
rho = round(cor(SAHA.response, HIV.response, method="spearman"), 2)
xx = cbind(SAHA.response, HIV.response)
set.seed(123)
boot = replicate(1000,
   cor(xx[sample(nrow(xx), replace=TRUE),], method="spearman")[1,2])
lo = round(sort(boot)[25], 2)
hi = round(sort(boot)[975], 2)
rhotext = paste("Spearman rho: ", rho,
   "\n95% bootstrap CI: (", lo, ", ", hi, ")", sep="", collapse="")
height = graphics::strheight(rhotext, cex=.8)
width = graphics::strwidth(rhotext, cex=.8)
rect(xleft=21, ybottom=8.8-height, xright=21+width, ytop=8.8,
     border=NA, col="white")
text(x=21, y=8.8, rhotext,
   family="Avenir Medium", col="gray40", adj=c(0,1), cex=.8)

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     # Change order to match the rest of the manuscript.
     col=type_colors[c(5,6)], pch=19, cex=.8,
     legend=c("J-Lat A2 + SAHA", "Jurkat + SAHA")
)

showtext_end()
dev.off()


PMA.response = PMA.response = t(t(PMA) - DMSO_x)
PMA.response = PMA.response %*% PMA_v

HIV.response = expr[type %in% c(3,4),ncol(expr)]

pdf("PMA-signature-vs-HIV.pdf", height=5, width=5)
showtext_begin()

par(mar=c(3.1,3.1,0,0))

plot(PMA.response, 100*HIV.response, panel.first=grid(),
   bty="n", ylab="", xlab="", xaxt="n", yaxt="n")
points(PMA.response, 100*HIV.response, pch=19, cex=.8,
   col=type_colors[type[type %in% c(3,4)]])

axis(side=1, col="gray30", cex.axis=.8, padj=-.9, col.axis="gray20")
axis(side=2, col="gray30", cex.axis=.8, padj= .9, col.axis="gray20")
title(xlab="Strength of the PMA signature (a.u.)", line=2, col.lab="gray30",
      family="Avenir Medium")
title(ylab="Reads in GFP-nef (%)", line=2, col.lab="gray30",
      family="Avenir Medium")

# Compute correlation with J-Lat A2 only (Jurkat have
# no HIV, so they artifically lower the correlation).
ptype = type[type %in% c(3,4)]
PMA.response = PMA.response[ptype == 3]
HIV.response = HIV.response[ptype == 3]
rho = round(cor(PMA.response, HIV.response, method="spearman"), 2)
xx = cbind(PMA.response, HIV.response)
set.seed(123)
boot = replicate(1000,
   cor(xx[sample(nrow(xx), replace=TRUE),], method="spearman")[1,2])
lo = round(sort(boot)[25], 2)
hi = round(sort(boot)[975], 2)
rhotext = paste("Spearman rho: ", rho,
   "\n95% bootstrap CI: (", lo, ", ", hi, ")", sep="", collapse="")
height = graphics::strheight(rhotext, cex=.8)
width = graphics::strwidth(rhotext, cex=.8)
rect(xleft=14.5, ybottom=3.4-height, xright=14.5+width, ytop=3.4,
     border=NA, col="white")
text(x=14.5, y=3.4, rhotext,
   family="Avenir Medium", col="gray40", adj=c(0,1), cex=.8)

legend(x="topright", inset=.01,
     bg="white", box.col="gray50",
     # Change order to match the rest of the manuscript.
     col=type_colors[c(3,4)], pch=19, cex=.8,
     legend=c("J-Lat A2 + PMA", "Jurkat + PMA")
)

showtext_end()
dev.off()
