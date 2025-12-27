# NetMedGPT - A network medicine foundation model for extensive disease mechanism mining and drug repurposing

## Overview
NetMedGPT is a transformer-based foundation model for network medicine that enables unified, zero-shot inference over large-scale biomedical knowledge graphs. The model learns contextualized representations of biomedical entities and relations via **masked token prediction on graph-derived sequences**.

Without task-specific retraining, NetMedGPT supports multiple biomedical inference tasks, including:
- Drug–disease indication prediction  
- Drug–target interaction prediction  
- Adverse drug reaction (ADR) prediction  
- Contraindication identification  
- Off-label use discovery 

In addition, NetMedGPT enables **scalable drug repurposing** and **mechanistic interpretation** through context-specific subnetwork generation.
It also includes an **interactive chatbot** that accepts free-text user queries, converts them into model-compatible pseudo-sentences, and returns ranked predictions.

<img width="2539" height="3235" alt="figure1_NetMedGPT_overview" src="https://github.com/user-attachments/assets/8c863158-8438-4862-a941-6b0b12a330ed" />

---

## Installation

### 1. Environment setup

Clone the repository and create a conda environment:

```bash
git clone https://github.com/faren-f/NetMedGPT.git
cd NetMedGPT

conda create -n netmedgpt python=3.10
conda activate netmedgpt

python -m pip install -U pip
pip install -r requirements.txt
pip install -e .












# installation
## Prepare environment

1. First clone the repository and install the environment
```
git clone https://github.com/faren-f/NetMedGPT.git
cd NetMedGPT
conda create -n netmedgpt python=3.10
conda activate netmedgpt
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

2. Download the data folder either from the barcode below:
<img width="164" height="164" alt="download" src="https://github.com/user-attachments/assets/376ee90b-04a7-45fe-be7f-e2c42eb6ee4f" />

or the link below:
```
wget --content-disposition "https://cloud.uni-hamburg.de/s/r74Ro8rmQ2sHwsL/download?accept=zip"
unzip *.zip
rm *.zip
```
and place in NetMedGPT

# NetMedGPT
## train 
NetMedGPT is evaluated in three sterategies of **random link split**, **zero shot split** and **disease area split**
To reproduce the results and save the correponding models run the command line below:

```
train.py 
```
To run the model and get the result use the command below




```
python netmedgpt_llm.py --user_text "user_text"
```
```"user_text"``` is the user query. For example
```
python netmedgpt_llm.py --user_text "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions"
```
The output is saved as a ```.csv``` file at ```data/user_response```.

## inference for drug repurposing
gives confidence score


## inference for all tasks


## subnetwork generation










