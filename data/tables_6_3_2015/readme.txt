table_1 :
   Used to generate
      - simple_nonzero_eco_roc_curve.png
      - simple_nonzero_eco_pre_recall_curve.png
   Generated using only commensal relationships with 30 groups of
   2 or 3 interacting species.  Each group has an interaction strength of 1.1x

table_2 :
   Used to generate
      - uniform_rare_eco_roc_curve.png
      - uniform_rare_eco_pre_recall_curve.png
   Generated using only commensal relationships with 30 groups of
   2 or 3 interacting species.  Each group has an interaction strength of 1.1x
   Used the simulated data from table 1 and rarified to 2000 individuals

table_3 :
   Used to generate
      - random_rare_eco_roc_curve.png
      - random_rare_eco_pre_recall_curve.png
   Generated using only commensal relationships with 30 groups of
   2 or 3 interacting species.  Each group has an interaction strength of 1.1x
   Used the simulated data from table 2 and rarified to a geometrically
   distributed sampling depth with a mininum of 2000 species.
   This is supposed to simulate a more realistic scenario of microbial sampling

table_4 :
   Used to generate table_5
   Only has two samples.  The first sample was taken from the pre-treatment collection
   in the PTSD study.  The second sample was taken from the post-treatment collection
   in the PTSD study.

table_5 :
   Generated from simsam.py in qiime using table_4
   Two clusters of related samples.  The first collection of samples is related
   to the pre-treatment sample.  The second collection of samples is related to the
   post-treatment sample.  The pre-treatment samples and the post-treatment samples
   should have a phylogenetic dissimilarity of 0.5
