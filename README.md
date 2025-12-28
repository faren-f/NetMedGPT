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
```

### 2. Data download

Download the `data/` directory using **one** of the following options.

#### Option A: QR code

<p align="center">
  <img src="https://github.com/user-attachments/assets/376ee90b-04a7-45fe-be7f-e2c42eb6ee4f" alt="Data download QR code" width="160"/>
</p>

### Option B: Command line

```bash
wget --content-disposition "https://cloud.uni-hamburg.de/s/r74Ro8rmQ2sHwsL/download?accept=zip"
unzip *.zip
rm *.zip
```

After downloading, place the extracted `data/` directory in the root of the NetMedGPT repository.

---

## Training and Evaluation Settings

NetMedGPT is evaluated under three complementary training and evaluation strategies:

- **Random link split**  
- **Zero-shot split**  
- **Disease area split**

Supported disease areas:
- adrenal_gland  
- anemia  
- autoimmune  
- cardiovascular  
- cell_proliferation  
- diabetes  
- mental_health  
- metabolic_disorder  
- neurodegenerative  

---

## Training Script Usage

The training script is executed from the command line using `train.py`.


### Command
python train.py [arguments]

### Arguments

- *--gpu*  
  GPU device ID  
  **Type:** int  
  **Default:** `0`

- *--seed*  
  Random seed for reproducibility  
  **Type:** int  
  **Default:** `1`

- *--inference*  
  Inference or data split strategy  
  **Type:** string  
  **Default:** `random_link_split`  

  **Allowed values:**
  - random_link_split
  - zero_shot_split
  - adrenal_gland
  - anemia
  - autoimmune
  - cardiovascular
  - cell_proliferation
  - diabetes
  - mental_health
  - metabolic_disorder
  - neurodigenerative

### Example
python train.py --gpu 1 --seed 1 --inference cardiovascular


### Notes

- If *--gpu* is not specified, GPU `0` is used.
- If *--seed* is not specified, the default seed is `1`.
- The value of *--inference* must be one of the listed options.

---

## Inference

### Drug repurposing inference

NetMedGPT produces ranked drug–disease predictions with associated confidence scores, enabling large-scale drug repurposing analyses without task-specific retraining.

### Multi-task inference

The pretrained model can be directly applied to all supported tasks, including drug–target prediction, ADR prediction, contraindication detection, and off-label use discovery.

---

## Subnetwork Generation

NetMedGPT supports context-specific subnetwork extraction for mechanistic interpretation. Given a query entity or task, the model generates informative subnetworks that highlight relevant biological and pharmacological pathways.












