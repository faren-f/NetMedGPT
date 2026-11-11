NetMedGPT retrained on the curated KG
=====================================

Question: does performance stay the same when NetMedGPT is trained on the curated KG
(nodes_preprocessed.csv, edges_preprocessed.csv, embeddings_preprocessedKG.pt: no drug_drug edges, drugs with broad
indications and misclassified drug-ADR edges removed)?

1. Curated splits (done; netmedgpt env):
       make_curated_splits.ipynb
   -> data/splits_curated/random_<seed>/ and complex_disease_<seed>/  (seeds 1-5), summary.csv
      train.csv, valid.csv and test.csv contain only the edges of the original split that are in the curated KG
      (the 1,477 removed drugs are not in any of them).

2. Training (train.py, --inference random_link_split_curated / zero_shot_split_curated; these options read the
   curated splits and the curated KG files, so the negatives are also sampled from the curated KG):
       cd /home/bbc8731/NetMedGPT
       for s in 1 2 3 4 5; do python train.py --gpu 0 --seed $s --inference random_link_split_curated; done
       for s in 1 2 3 4 5; do python train.py --gpu 0 --seed $s --inference zero_shot_split_curated; done
   -> checkpoints data/model_checkpoints/saved_models_evaluation/netmedgpt_<inference>_seed<s>.pt
   -> results    data/result/netmedgpt_<inference>_seed<s>/
   (the original checkpoints and results are not touched)

3. Comparison on identical test pairs (after training):
       compare_curated.ipynb
   Builds the curated test pairs (positives + negatives, as in other/evaluate_checkpoints.py), scores the original
   checkpoints (pairs mapped to full-KG indices by node ID) and the curated checkpoints on them; AUROC, AUPRC,
   Recall@100 per relation, paired t-test over seeds. Seeds without a curated checkpoint are skipped.
   -> data/new_analysis/curated_kg_training/curated_vs_original_{per_seed,summary}.csv

Note: on the curated test pairs the original model scores lower than on the full test set (e.g. random split seed 1,
indication AUROC 0.894 vs 0.947), because the removed drugs - frequent, easy-to-predict drugs - are not in these pairs.
The comparison is therefore always against the original model on the same curated pairs, not against the paper values.
