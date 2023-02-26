SCHIV= --rm -v $$(pwd):/tmp -u$$(id -u):$$(id -g) --privileged schiv
all:
	echo "select target"

# Remove genes with low expression.
allcells.tsv: data/exprMatrix.tsv data/sampleSheet.tsv
	docker run $(SCHIV) Rscript scripts/generate-table.R

# Plot PCA and total reads to remove "dead" cells.
# File PC1_up_gene_ids.txt is fed to the DAVID web site.
pca-all-cells.pdf correlates.pdf all-cells-total-reads.pdf PC1_up_gene_ids.txt: allcells.tsv
	docker run $(SCHIV) Rscript scripts/plot-pca-all-cells.R
# File DAVID_PC1.txt is generated manually from the DAVID web site.
barplot-GO-PC1.pdf: DAVID_PC1.txt
	docker run $(SCHIV) Rscript scripts/show-GO.R $< $@

# Remove "dead cells" using PCA.
alivecells.tsv: allcells.tsv
	docker run $(SCHIV) Rscript scripts/make-alive-cells.R

# Create cell subsets.
alivecells_wo_HIV.tsv: alivecells.tsv
	docker run $(SCHIV) Rscript scripts/make-alive-cells-wo-HIV.R
aliveJLat_SAHA.tsv: alivecells.tsv
	docker run $(SCHIV) Rscript scripts/filter_alive_cells.R J-LatA2+SAHA $@
aliveJLat_PMA.tsv: alivecells.tsv
	docker run $(SCHIV)  Rscript scripts/filter_alive_cells.R J-LatA2+PMA $@

# Plot PCA with alive cells only.
# Files SAHA_up_gene_ids.txt and PMA_up_gene_ids.txt are fed to the DAVID web site.
pca-alive-cells.pdf SAHA_up_gene_ids.txt PMA_up_gene_ids.txt SAHA-signature-vs-HIV.pdf PMA-signature-vs-HIV.pdf: alivecells.tsv
	docker run $(SCHIV) Rscript scripts/plot-pca-alive-cells.R
# File DAVID_SAHA.txt is generated manually from the DAVID web site.
barplot-GO-SAHA.pdf: DAVID_SAHA.txt
	docker run $(SCHIV) Rscript scripts/show-GO.R $< $@
# File DAVID_PMA.txt is generated manually from the DAVID web site.
barplot-GO-PMA.pdf: DAVID_PMA.txt
	docker run $(SCHIV) Rscript scripts/show-GO.R $< $@

# Perform BB and LDA with 3 groups (without HIV).
K3_BB.pt: alivecells_wo_HIV.tsv 
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 3 $< $@
K3_LDA.pt: alivecells_wo_HIV.tsv 
	docker run --gpus all $(SCHIV) python scripts/lda.py 3 $< $@

# Plot module simplex.
K3_posterior_theta_BB.tsv: K3_BB.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_theta
K3_posterior_theta_LDA.tsv: K3_LDA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param doc_topic_posterior
simplex_BB.pdf: K3_posterior_theta_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot_simplex.R $< $@
simplex_LDA.pdf: K3_posterior_theta_LDA.tsv
	docker run $(SCHIV) Rscript scripts/plot_simplex.R $< $@

# Plot signature vs HIV <==== NEED TO WORK ON THAT BIT.
hello.pdf: K3_posterior_theta_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot_signature_vs_HIV.R $< $@


# Posterior predictive of MT-CO1 (most expressed gene).
MTCO1_BB.tsv: K3_BB.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4633
MTCO1_LDA.tsv: K3_LDA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4633
posterior_predictive_MTCO1_BB.pdf: MTCO1_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot-MTCO1.R $< $@
posterior_predictive_MTCO1_LDA.pdf: MTCO1_LDA.tsv
	docker run $(SCHIV) Rscript scripts/plot-MTCO1.R $< $@


# Perform BB for separate treatments with 2 groups...
K2_SAHA.pt: aliveJLat_SAHA.tsv
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 2 $< $@
K2_PMA.pt: aliveJLat_PMA.tsv 
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 2 $< $@
# ... and with 3 groups.
K3_SAHA.pt: aliveJLat_SAHA.tsv 
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 3 $< $@
K3_PMA.pt: aliveJLat_PMA.tsv 
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 3 $< $@
K3_SAHA_posterior_theta.tsv: K3_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_theta
K3_PMA_posterior_theta.tsv: K3_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_theta


# Posterior predictive of HIV
K2_HIV_SAHA_BB.tsv: K2_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4951
posterior_predictive_K2_HIV_SAHA_BB.pdf: K2_HIV_SAHA_BB.tsv aliveJLat_SAHA.tsv
	docker run $(SCHIV) Rscript scripts/plot_HIV.R $^ $@
K2_HIV_PMA_BB.tsv: K2_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4951
posterior_predictive_K2_HIV_PMA_BB.pdf: K2_HIV_PMA_BB.tsv aliveJLat_PMA.tsv
	docker run $(SCHIV) Rscript scripts/plot_HIV.R $^ $@
K3_HIV_SAHA_BB.tsv: K3_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4951
posterior_predictive_K3_HIV_SAHA_BB.pdf: K3_HIV_SAHA_BB.tsv aliveJLat_SAHA.tsv
	docker run $(SCHIV) Rscript scripts/plot_HIV.R $^ $@
K3_HIV_PMA_BB.tsv: K3_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ gene_sample 4951
posterior_predictive_K3_HIV_PMA_BB.pdf: K3_HIV_PMA_BB.tsv aliveJLat_PMA.tsv
	docker run $(SCHIV) Rscript scripts/plot_HIV.R $^ $@


## Plot representation of drug signatures.
#bargraph-alive-cells-topics-no-bec-no-HIV.pdf simplex-topics-alive-cells-no-bec-no-HIV.pdf SAHA-signature-vs-HIV.pdf PMA-signature-vs-HIV.pdf: alivecells.tsv out-alive-no-bec-no-HIV.txt
#	$(DOCKER_CPU) Rscript scripts/plot-topics-alive-cells.R

# Print genes in signatures for pathway analysis.
#PMA-weights.csv SAHA-weights.csv DMSO-weights.csv PMA-gene_ids.txt SAHA-gene_ids.txt DMSO-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-alive-no-bec-no-HIV.txt
#	$(DOCKER_CPU) Rscript scripts/make-word-signatures-alive-cells.R

## The files SAHA-DAVID.txt and PMA-DAVID.txt are generated
## manually from the DAVID web site.
#PMA-GO.txt SAHA-GO.txt barplot-GO-PMA.pdf barplot-GO-SAHA.pdf: SAHA-DAVID.txt PMA-DAVID.txt
#	$(DOCKER_CPU) Rscript scripts/show-GO.R


# Distribution of the LDA modules.
K2_SAHA_posterior_theta_BB.tsv: K2_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_theta
K2_PMA_posterior_theta_BB.tsv: K2_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_theta
bargraph-modules-SAHA.pdf: K2_SAHA_posterior_theta_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot-topics.R aliveJLat_SAHA.tsv $< $@
bargraph-modules-PMA.pdf: K2_PMA_posterior_theta_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot-topics.R aliveJLat_PMA.tsv $< $@


# Module profiles.
K2_posterior_g_SAHA.tsv: K2_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_g_loc
K2_posterior_g_PMA.tsv: K2_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_g_loc
K3_posterior_g_SAHA.tsv: K3_SAHA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_g_loc
K3_posterior_g_PMA.tsv: K3_PMA.pt
	docker run $(SCHIV) python scripts/extract_blackbox.py $< $@ param posterior_g_loc
K2_modules_SAHA: K2_posterior_g_SAHA.tsv
	docker run $(SCHIV) Rscript scripts/print_modules.R $< $@
K2_modules_PMA: K2_posterior_g_PMA.tsv
	docker run $(SCHIV) Rscript scripts/print_modules.R $< $@
K3_modules_PMA: K3_posterior_g_PMA.tsv
	docker run $(SCHIV) Rscript scripts/print_modules.R $< $@
K3_modules_SAHA: K3_posterior_g_SAHA.tsv
	docker run $(SCHIV) Rscript scripts/print_modules.R $< $@

################################

## LDA modules vs HIV.
#topics-SAHA-vs-HIV.pdf: out-SAHA-2.txt
#	$(DOCKER_CPU) Rscript scripts/plot-stripes.R $< $@ TRUE
#topics-PMA-vs-HIV.pdf: out-PMA-2.txt
#	$(DOCKER_CPU) Rscript scripts/plot-stripes.R $< $@ TRUE
#
## Control LDA with scramble HIV
#scramble-traces-PMA.txt: alivecells.tsv
#	$(DOCKER_CPU) python scripts/LDA-2-groups-PMA-scramble.py > $@
#scramble-traces-SAHA.txt: alivecells.tsv
#	$(DOCKER_CPU) python scripts/LDA-2-groups-SAHA-scramble.py > $@
#
#
## Scramble traces (SAHA + PMA)
#scramble-traces.pdf: scramble-traces-SAHA.txt scramble-traces-PMA.txt
#	$(DOCKER_CPU) Rscript scripts/plot-scramble-traces.R
#
### Plot signature proportions with PMA.
##bargraph-topics-PMA.pdf bargraph-topics-PMA-no-bec.pdf: alivecells.tsv out-PMA.txt out-PMA-no-bec.txt
##	$(DOCKER_CPU) Rscript scripts/plot-topics-PMA.R
##
### Plot signature proportions with SAHA.
##bargraph-topics-SAHA.pdf bargraph-topics-SAHA-no-bec.pdf: alivecells.tsv out-SAHA.txt out-SAHA-no-bec.txt
##	$(DOCKER_CPU) Rscript scripts/plot-topics-SAHA.R
##
### Print genes in signatures for pathway analysis.
##SAHA-signature-weights.csv SAHA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-SAHA.txt
##	$(DOCKER_CPU) Rscript scripts/make-word-signatures-SAHA.R
##
### Print genes in signatures for pathway analysis.
##PMA-signature-weights.csv PMA-signature-gene_ids.txt: data/gene_annotations.tsv alivecells.tsv wfreq-PMA.txt
##	$(DOCKER_CPU) Rscript scripts/make-word-signatures-PMA.R
##
### Print genes in signature for pathway analysis.
##PMA-signature-GO.txt SAHA-signature-GO.txt barplot-GO-PMA-signature.pdf  barplot-GO-SAHA-signature.pdf: SAHA-signature-DAVID.txt PMA-signature-DAVID.txt
##	$(DOCKER_CPU) Rscript scripts/show-GO-signatures.R
##
### Compare signatures of Jurkat and J-LatA2 cells.
##scatter-PMA-signature.pdf scatter-SAHA-signature.pdf: wfreq-PMA.txt wfreq-SAHA.txt type-effects-PMA.txt type-effects-SAHA.txt
##	Rscript plot-scatter-signatures.R
#
############## EXTRAS #############
#
## Perform basic LDA, with HIV (for the record).
#out-alive-no-bec.txt wfreq-alive-no-bec.txt: alivecells.tsv
#	$(DOCKER_CPU) python scripts/LDA-3-groups-alive-cells-no-bec.py
