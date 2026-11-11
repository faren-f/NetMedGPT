NetMedGPT retrained on the curated KG
=====================================

Question: does performance stay the same when NetMedGPT is trained only on the edges of the curated KG
(edges_preprocessed.csv: no drug_drug edges, drugs with broad indications and misclassified drug-ADR edges removed)?

1. Curated training splits (done; netmedgpt env, from this folder or the repository root):
       make_curated_splits.ipynb
   -> data/splits_curated/random_<seed>/ and complex_disease_<seed>/  (seeds 1-5)
      train.csv = training edges of the original split that are in the curated KG (66% of the rows; drug_drug and the
      edges of the 1,477 removed drugs are dropped); valid.csv / test.csv = unchanged copies, so the test pairs
      (including the negatives) are identical to the original runs.
   -> data/splits_curated/summary.csv

2. Training (train.py, new --inference options random_link_split_curated / zero_shot_split_curated):
       cd /home/bbc8731/NetMedGPT
       for s in 1 2 3 4 5; do python train.py --gpu 0 --seed $s --inference random_link_split_curated; done
       for s in 1 2 3 4 5; do python train.py --gpu 0 --seed $s --inference zero_shot_split_curated; done
   -> checkpoints data/model_checkpoints/saved_models_evaluation/netmedgpt_<inference>_seed<s>.pt
   -> results    data/result/netmedgpt_<inference>_seed<s>/
   (the original checkpoints and results are not touched)

3. Comparison with the original runs (to do after training): same test pairs, paired over seeds.
   Report two versions:
   - all test pairs;
   - test pairs whose drug is in the curated KG. About half of the indication edges (and ~23% of all test drug edges)
     involve the 1,477 drugs removed by curation; in the curated runs these drugs have no training edges, so a drop on
     those pairs is expected and is not an effect of the pseudo-sentence coverage.
