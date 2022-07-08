DOCKER_RUN= docker run --rm -v $$(pwd):/tmp -u$$(id -u):$$(id -g) schiv
all:
	echo "select target"

# Remove genes with low expression.
allcells.tsv: data/exprMatrix.tsv data/sampleSheet.tsv
	$(DOCKER_RUN) Rscript scripts/generate-table.R

# Plot PCA and total reads to remove "dead" cells.
# File PC1_up_gene_ids.txt is fed to the DAVID web site.
pca-all-cells.pdf correlates.pdf all-cells-total-reads.pdf PC1_up_gene_ids.txt: allcells.tsv
	$(DOCKER_RUN) Rscript scripts/plot-pca-all-cells.R
# File DAVID_PC1.txt is generated manually from the DAVID web site.
barplot-GO-PC1.pdf: DAVID_PC1.txt
	$(DOCKER_RUN) Rscript scripts/show-GO.R $< $@

# Remove "dead cells" using PCA.
alivecells.tsv: allcells.tsv
	$(DOCKER_RUN) Rscript scripts/make-alive-cells.R

# Plot PCA with alive cells only.
# Files SAHA_up_gene_ids.txt and PMA_up_gene_ids.txt are fed to the DAVID web site.
pca-alive-cells.pdf SAHA_up_gene_ids.txt PMA_up_gene_ids.txt SAHA-signature-vs-HIV.pdf PMA-signature-vs-HIV.pdf: alivecells.tsv
	$(DOCKER_RUN) Rscript scripts/plot-pca-alive-cells.R
# File DAVID_SAHA.txt is generated manually from the DAVID web site.
barplot-GO-SAHA.pdf: DAVID_SAHA.txt
	$(DOCKER_RUN) Rscript scripts/show-GO.R $< $@
# File DAVID_PMA.txt is generated manually from the DAVID web site.
barplot-GO-PMA.pdf: DAVID_PMA.txt
	$(DOCKER_RUN) Rscript scripts/show-GO.R $< $@

## Perform basic LDA with 3 groups, without HIV.
#out-alive-no-bec-no-HIV.txt wfreq-alive-no-bec-no-HIV.txt: alivecells.tsv
#	$(DOCKER_RUN) python scripts/LDA-3-groups-alive-cells-no-bec-no-HIV.py

## Plot representation of drug signatures.
#bargraph-alive-cells-topics-no-bec-no-HIV.pdf simplex-topics-alive-cells-no-bec-no-HIV.pdf SAHA-signature-vs-HIV.pdf PMA-signature-vs-HIV.pdf: alivecells.tsv out-alive-no-bec-no-HIV.txt
#	$(DOCKER_RUN) Rscript scripts/plot-topics-alive-cells.R

# Print genes in signatures for pathway analysis.
PMA-weights.csv SAHA-weights.csv DMSO-weights.csv PMA-gene_ids.txt SAHA-gene_ids.txt DMSO-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-alive-no-bec-no-HIV.txt
	$(DOCKER_RUN) Rscript scripts/make-word-signatures-alive-cells.R

## The files SAHA-DAVID.txt and PMA-DAVID.txt are generated
## manually from the DAVID web site.
#PMA-GO.txt SAHA-GO.txt barplot-GO-PMA.pdf barplot-GO-SAHA.pdf: SAHA-DAVID.txt PMA-DAVID.txt
#	$(DOCKER_RUN) Rscript scripts/show-GO.R

# LDA without batch-effect correction.
out-SAHA-no-bec.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-SAHA-no-bec.py
out-PMA-no-bec.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-PMA-no-bec.py
bargraph-topics-SAHA-no-bec.pdf: out-SAHA-no-bec.txt
	$(DOCKER_RUN) Rscript scripts/plot-topics.R $< $@ FALSE
bargraph-topics-PMA-no-bec.pdf: out-PMA-no-bec.txt
	$(DOCKER_RUN) Rscript scripts/plot-topics.R $< $@ FALSE


# LDA with batch-effect correction.
out-SAHA-2.txt wfreq-SAHA-2.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-SAHA.py
out-PMA-2.txt wfreq-PMA-2.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-PMA.py
bargraph-topics-SAHA.pdf: out-SAHA-2.txt
	$(DOCKER_RUN) Rscript scripts/plot-topics.R $< $@ FALSE
bargraph-topics-PMA.pdf: out-PMA-2.txt
	$(DOCKER_RUN) Rscript scripts/plot-topics.R $< $@ TRUE

# Module profiles.
modules-SAHA.pdf modules-PMA.pdf: wfreq-SAHA-2.txt wfreq-PMA-2.txt
	$(DOCKER_RUN) Rscript scripts/plot-modules.R


# Control LDA with scramble HIV
scramble-traces-PMA.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-PMA-scramble.py > $@
scramble-traces-SAHA.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-2-groups-SAHA-scramble.py > $@


# Scramble traces (SAHA + PMA)
scramble-traces.pdf: scramble-traces-SAHA.txt scramble-traces-PMA.txt
	$(DOCKER_RUN) Rscript scripts/plot-scramble-traces.R

## Plot signature proportions with PMA.
#bargraph-topics-PMA.pdf bargraph-topics-PMA-no-bec.pdf: alivecells.tsv out-PMA.txt out-PMA-no-bec.txt
#	$(DOCKER_RUN) Rscript scripts/plot-topics-PMA.R
#
## Plot signature proportions with SAHA.
#bargraph-topics-SAHA.pdf bargraph-topics-SAHA-no-bec.pdf: alivecells.tsv out-SAHA.txt out-SAHA-no-bec.txt
#	$(DOCKER_RUN) Rscript scripts/plot-topics-SAHA.R
#
## Print genes in signatures for pathway analysis.
#SAHA-signature-weights.csv SAHA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-SAHA.txt
#	$(DOCKER_RUN) Rscript scripts/make-word-signatures-SAHA.R
#
## Print genes in signatures for pathway analysis.
#PMA-signature-weights.csv PMA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-PMA.txt
#	$(DOCKER_RUN) Rscript scripts/make-word-signatures-PMA.R
#
## Print genes in signature for pathway analysis.
#PMA-signature-GO.txt SAHA-signature-GO.txt barplot-GO-PMA-signature.pdf  barplot-GO-SAHA-signature.pdf: SAHA-signature-DAVID.txt PMA-signature-DAVID.txt
#	$(DOCKER_RUN) Rscript scripts/show-GO-signatures.R
#
## Compare signatures of Jurkat and J-LatA2 cells.
#scatter-PMA-signature.pdf scatter-SAHA-signature.pdf: wfreq-PMA.txt wfreq-SAHA.txt type-effects-PMA.txt type-effects-SAHA.txt
#	Rscript plot-scatter-signatures.R

############# EXTRAS #############

# Perform basic LDA, with HIV (for the record).
out-alive-no-bec.txt wfreq-alive-no-bec.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-alive-cells-no-bec.py
