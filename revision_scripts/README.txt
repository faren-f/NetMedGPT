revision_scripts/ — scripts written for the revision (reviewer responses)
=====================================================================

All of these are new; nothing of the original pipeline (train.py, config.py, generate_subnetwork.py,
inference_*.py, netmedgpt/) was moved or changed.

How to run
----------
Always from the repository root (/home/bbc8731/NetMedGPT), with the root on PYTHONPATH:

    cd /home/bbc8731/NetMedGPT
    PYTHONPATH=. python revision_scripts/<script>.py ...

PYTHONPATH=. is needed because the scripts import config.py and the netmedgpt/ package, which stay in the
repository root. The data paths (data/KG/, data/splits/, data/result/) are relative to the root as well, so the
working directory must be the root.

Shared helpers
--------------
common.py        splits, cached walk corpus and val/test pairs, metrics, saving, scorer adapters
baseline_utils.py  relation tokens, evaluation + saving wrapper, DREAMwalk helpers
memlimit.py      RAM watchdog (kills a job above --limit_gb; writes memlimit.log)

Walk-embedding baselines (reviewer comment 2)   -> how_to_run_walk_baselines.txt
---------------------------------------------
build_corpus.py                    walk corpus + val/test pairs (cached per split and seed)
node2vec_baseline.py               4 variants: +-relation tokens x cosine / logistic regression
dreamwalk_samecorpus.py            DREAMwalk skip-gram + XGBoost on the identical corpus
dreamwalk_native.py                DREAMwalk with its own walks (not recommended, see the how-to)
run_all_baselines.sh               runs the above for both splits and seeds 1-5
run_baselines.sh                   older version of the same loop

Strong recent baselines (reviewer comment 3)
--------------------------------------------
biopathnet_baseline.py   BioPathNet (Nature BME 2026), both splits        -> /home/bbc8731/BioPathNet/how_to_run.txt
psichic_baseline.py      PSICHIC (Nat Mach Intell 2024), DTI              -> /home/bbc8731/PSICHIC/how_to_run.txt
deepadr_baseline.py      DeepADR (Brief Bioinform), ADR                   -> /home/bbc8731/DeepADR/how_to_run.txt
galeano_baseline.py      Galeano et al. (Nat Commun 2020), ADR            -> /home/bbc8731/Side-effect-Frequencies/how_to_run.txt
dti_inputs.py            drug SMILES (PubChem) + protein sequences (UniProt); used by PSICHIC and DeepADR
adr_inputs.py            MoLFormer drug + BioBERT phenotype embeddings; used by DeepADR

Statistics and paired comparisons
---------------------------------
statistical_significance_walk_baselines.py   NetMedGPT vs every method in walk_baselines_long.csv
export_netmedgpt_scores.py                   per-pair NetMedGPT scores + per-drug Recall@100 (saved checkpoints)
compare_overlap.py                           NetMedGPT vs a baseline on identical pairs (generic, e.g. galeano)
compare_overlap_deepadr.py                   NetMedGPT vs DeepADR (all pairs and overlap)
compare_overlap_psichic.py                   NetMedGPT vs PSICHIC (all pairs and overlap)

Walk coverage notebooks (reviewer comment 2)
--------------------------------------------
walk_coverage_analysis.ipynb, walk_coverage_baselines.ipynb, walk_coverage_curated_kg.ipynb,
walk_coverage_results.ipynb   what graph information the random walks keep and lose.
Each starts with a "path setup" cell that changes the working directory back to the repository root and puts the
root and this folder on sys.path, so the data paths and the config / netmedgpt imports work unchanged. Re-running
that cell is harmless; delete it if a notebook is moved back to the root.

Interpretability (reviewer comment on interpretability)
------------------------------------------------------
subnetwork_uncertainty.py   temperature sweep + MC-dropout uncertainty per subnetwork edge
                            -> ../explanations/interpretability_uncertainty_plan.txt
                            -> ../explanations/subnetwork_explorer/ (interactive page)

Outputs
-------
data/new_analysis/     intermediate files (corpus cache, baseline inputs, features, per-pair scores)
data/result/new_analysis/   per-run result CSVs, walk_baselines_long.csv, overlap comparisons

Temporary copies in the repository root
---------------------------------------
memlimit.py, psichic_baseline.py, common.py and baseline_utils.py also exist in the repository root as copies,
because the PSICHIC seed loop started before this folder was created and launches its remaining seed from the root.
Delete those four root copies once that loop has finished; the versions here are the ones to keep.
