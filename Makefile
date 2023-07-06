SCHIV= --rm -v $$(pwd):/tmp -u$$(id -u):$$(id -g) --privileged schiv
all:
	echo "select target"

# Remove genes with low expression.
allcells.tsv: data/exprMatrix.tsv data/sampleSheet.tsv
	docker run $(SCHIV) Rscript scripts/generate-table.R

# Plot PCA and total reads to remove "dead" cells.
# File PC1_up_gene_ids.txt is fed to the DAVID web site.
pca_all_cells.pdf correlates.pdf all-cells-total-reads.pdf PC1_up_gene_ids.txt: allcells.tsv
	docker run $(SCHIV) Rscript scripts/plot_pca_all_cells.R
# File DAVID_PC1.txt is generated manually from the DAVID web site.
barplot-GO-PC1.pdf: DAVID_PC1.txt
	docker run $(SCHIV) Rscript scripts/show-GO.R $< $@

# Remove "dead cells" using PCA.
alivecells.tsv: allcells.tsv
	docker run $(SCHIV) Rscript scripts/make-alive-cells.R

## Create cell subsets.
alivecells_SAHA.tsv: alivecells.tsv
	docker run $(SCHIV) Rscript scripts/filter_alive_cells.R "SAHA" $@
alivecells_PMA.tsv: alivecells.tsv
	docker run $(SCHIV) Rscript scripts/filter_alive_cells.R "PMA" $@
# Remove HIV (e.g., alivecells_no_HIV.tsv).
%_no_HIV.tsv: %.tsv
	docker run $(SCHIV) true && awk -v OFS="\t" "NF{NF-=1};1" $< > $@

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

# Perform blackbox (BB) and LDA
K2_SAHA_LDA.pt: alivecells_SAHA.tsv
	docker run --gpus all $(SCHIV) python scripts/lda.py 2 $< $@
K2_PMA_LDA.pt: alivecells_PMA.tsv
	docker run --gpus all $(SCHIV) python scripts/lda.py 2 $< $@
K2_BB.pt: alivecells.tsv
	docker run --gpus all $(SCHIV) python scripts/blackbox.py 2 $< $@
# Control blackbox with multinomial a distribution.
K2_BB_multinom.pt: alivecells.tsv
	docker run --gpus all $(SCHIV) python scripts/blackbox_multinom.py 2 $< $@

# Breakdown of modules A/B.
K2_SAHA_LDA_post_theta_param.tsv: K2_SAHA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_theta_param
K2_PMA_LDA_post_theta_param.tsv: K2_PMA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_theta_param
K2_BB_post_theta_param.tsv: K2_BB.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_theta_loc
K2_SAHA_LDA_theta.pdf: alivecells_SAHA.tsv K2_SAHA_LDA_post_theta_param.tsv
	docker run $(SCHIV) Rscript scripts/plot_theta.R $^ $@ all
K2_PMA_LDA_theta.pdf: alivecells_PMA.tsv K2_PMA_LDA_post_theta_param.tsv
	docker run $(SCHIV) Rscript scripts/plot_theta.R $^ $@ all
K2_SAHA_BB_theta.pdf: alivecells.tsv K2_BB_post_theta_param.tsv
	docker run $(SCHIV) Rscript scripts/plot_theta.R $^ $@ SAHA
K2_PMA_BB_theta.pdf: alivecells.tsv K2_BB_post_theta_param.tsv
	docker run $(SCHIV) Rscript scripts/plot_theta.R $^ $@ PMA

# Posterior check for MT-CO1 (most expressed gene).
MTCO1_SAHA_LDA.tsv: K2_SAHA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ gene_sample 4633
MTCO1_PMA_LDA.tsv: K2_PMA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ gene_sample 4633
MTCO1_BB.tsv: K2_BB.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ gene_sample 4633
MTCO1_BB_multinom.tsv: K2_BB_multinom.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ gene_sample 4633
post_check_MTCO1_SAHA_LDA.pdf: alivecells_SAHA.tsv MTCO1_SAHA_LDA.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ all
post_check_MTCO1_PMA_LDA.pdf: alivecells_PMA.tsv MTCO1_PMA_LDA.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ all
post_check_MTCO1_SAHA_BB.pdf: alivecells.tsv MTCO1_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ SAHA
post_check_MTCO1_PMA_BB.pdf: alivecells.tsv MTCO1_BB.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ PMA
post_check_MTCO1_SAHA_BB_multinom.pdf: alivecells.tsv MTCO1_BB_multinom.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ SAHA
post_check_MTCO1_PMA_BB_multinom.pdf: alivecells.tsv MTCO1_BB_multinom.tsv
	docker run $(SCHIV) Rscript scripts/plot_MTCO1.R $^ $@ PMA


## Plot module simplexes.
#K3_LDA_post_theta.tsv: K3_LDA.pt
#	docker run $(SCHIV) python scripts/extract.py $< $@ param doc_topic_posterior
#K2_post_theta_BB.tsv: K2_BB.pt
#	docker run $(SCHIV) python scripts/extract.py $< $@ param post_theta_loc
#simplex_LDA.pdf: K2_post_theta_LDA.tsv alivecells_no_HIV.tsv
#	docker run $(SCHIV) Rscript scripts/plot_simplex.R $^ $@
#proj_BB.pdf: K2_post_theta_BB.tsv alivecells.tsv
#	docker run $(SCHIV) Rscript scripts/plot_multi_proj.R $^ $@

# Profiles.
K2_SAHA_LDA_post_phi_param.tsv: K2_SAHA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_phi_param
K2_PMA_LDA_post_phi_param.tsv: K2_PMA_LDA.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_phi_param
K2_BB_post_base_loc.tsv: K2_BB.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_base_loc
K2_BB_post_mod_loc.tsv: K2_BB.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ param post_mod_loc

# Show / print module profiles.
K2_SAHA_LDA: K2_SAHA_LDA_post_phi_param.tsv
	docker run $(SCHIV) Rscript scripts/show_modules.R $< paired
K2_BB_types: K2_BB_post_base_loc.tsv
	docker run $(SCHIV) Rscript scripts/show_modules.R $< grouped
K2_BB_batches: K2_BB_post_base_loc.tsv
	docker run $(SCHIV) Rscript scripts/show_modules.R $< variance
K2_BB_modules: K2_BB_post_mod_loc.tsv
	docker run $(SCHIV) Rscript scripts/show_modules.R $< paired

# Hello
sm.tsv: K2_BB.pt
	docker run $(SCHIV) python scripts/extract.py $< $@ sm
pca_sm.pdf: sm.tsv
	docker run $(SCHIV) Rscript scripts/plot_pca_sm.R $< $@

################################

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
