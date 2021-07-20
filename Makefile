DOCKER_RUN= docker run --rm -v $$(pwd):/tmp -u$$(id -u):$$(id -g) schiv
all:
	echo "select target"

# Remove genes with low expression.
allcells.tsv: data/exprMatrix.tsv data/sampleSheet.tsv
	$(DOCKER_RUN) R -f scripts/generate-table.R

# Plot PCA and total reads to remove "dead" cells.
pca-all-cells.pdf all-cells-total-reads.pdf: allcells.tsv
	$(DOCKER_RUN) R -f scripts/plot-pca-all-cells.R

# Remove dead cells using PCA.
alivecells.tsv: allcells.tsv
	$(DOCKER_RUN) R -f scripts/make-alive-cells.R

# Perform basic LDA, without HIV.
out-alive-no-bec-no-HIV.txt wfreq-alive-no-bec-no-HIV.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-alive-cells-no-bec-no-HIV.py

# Plot representation of drug signatures.
bargraph-alive-cells-topics-no-bec-no-HIV.pdf simplex-topics-alive-cells-no-bec-no-HIV.pdf SAHA-signature-vs-HIV.pdf PMA-signature-vs-HIV.pdf: alivecells.tsv out-alive-no-bec-no-HIV.txt
	$(DOCKER_RUN) R -f scripts/plot-topics-alive-cells.R

# Print genes in signatures for pathway analysis.
PMA-weights.csv SAHA-weights.csv PMA-gene_ids.txt SAHA-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-alive-no-bec-no-HIV.txt
	$(DOCKER_RUN) R -f scripts/make-word-signatures-alive-cells.R

# The files SAHA-DAVID.txt and PMA-DAVID.txt are generated
# manually from the DAVID web site.
PMA-GO.txt SAHA-GO.txt barplot-GO-PMA.pdf barplot-GO-SAHA.pdf: SAHA-DAVID.txt PMA-DAVID.txt
	$(DOCKER_RUN) R -f scripts/show-GO.R

# LDA with batch-effect correction.
out-PMA.txt wfreq-PMA.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-PMA.py

# LDA without batch-effect correction.
out-PMA-no-bec.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-PMA-no-bec.py

# LDA with batch-effect correction and no HIV.
out-PMA-no-HIV.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-PMA-no-HIV.py

# LDA with batch-effect correction.
out-SAHA.txt wfreq-SAHA.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-SAHA.py

# LDA without batch-effect correction.
out-SAHA-no-bec.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-SAHA-no-bec.py

# LDA with batch-effect correction and no HIV.
out-SAHA-no-HIV.txt: alivecells.tsv
	$(DOCKER_RUN) python scripts/LDA-3-groups-SAHA-no-HIV.py

# Plot signature proportions with PMA.
bargraph-topics-PMA.pdf bargraph-topics-PMA-no-bec.pdf: alivecells.tsv out-PMA.txt out-PMA-no-bec.txt
	$(DOCKER_RUN) R -f scripts/plot-topics-PMA.R

# Plot signature proportions with SAHA.
bargraph-topics-SAHA.pdf bargraph-topics-SAHA-no-bec.pdf: alivecells.tsv out-SAHA.txt out-SAHA-no-bec.txt
	$(DOCKER_RUN) R -f scripts/plot-topics-SAHA.R

# Print genes in signatures for pathway analysis.
SAHA-signature-weights.csv SAHA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-SAHA.txt
	$(DOCKER_RUN) R -f scripts/make-word-signatures-SAHA.R

# Print genes in signatures for pathway analysis.
PMA-signature-weights.csv PMA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-PMA.txt
	$(DOCKER_RUN) R -f scripts/make-word-signatures-PMA.R

# Print genes in signature for pathway analysis.
PMA-signature-GO.txt SAHA-signature-GO.txt barplot-GO-PMA-signature.pdf  barplot-GO-SAHA-signature.pdf: SAHA-signature-DAVID.txt PMA-signature-DAVID.txt
	$(DOCKER_RUN) R -f scripts/show-GO-signatures.R

# Compare signatures of Jurkat and J-LatA2 cells.
scatter-PMA-signature.pdf scatter-SAHA-signature.pdf: wfreq-PMA.txt wfreq-SAHA.txt type-effects-PMA.txt type-effects-SAHA.txt
	R -f plot-scatter-signatures.R
