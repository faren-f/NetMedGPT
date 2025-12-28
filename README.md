# NetMedGPT - A network medicine foundation model for extensive disease mechanism mining and drug repurposing

## Overview
NetMedGPT is a transformer-based foundation model for network medicine that enables unified, zero-shot inference over large-scale biomedical knowledge graphs. The model learns contextualized representations of biomedical entities and relations via **masked token prediction on graph-derived sequences**.

Without task-specific retraining, NetMedGPT supports multiple biomedical inference tasks, including:
- Drug–disease indication prediction
- Drug–disease Contraindication prediction  
- Drug–target interaction prediction  
- Adverse drug reaction (ADR) prediction  
- Drug–disease Off-label use prediction 

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

#### Option B: Command line

```bash
wget --content-disposition "https://cloud.uni-hamburg.de/s/r74Ro8rmQ2sHwsL/download?accept=zip"
unzip data.zip
rm data.zip
```

After downloading, place the extracted `data/` directory in the root of the NetMedGPT repository.

---

## Training and Evaluation Settings

NetMedGPT is assessed under three evaluation strategies:

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

### Training

The training script is executed from the command line using `train.py`.


#### Command

python train.py [arguments]


#### Arguments

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

#### Example
```
python train.py --gpu 1 --seed 1 --inference cardiovascular
```

---

## Inference

### Drug repurposing inference

Drug repurposing inference proritizes candidate drugs for a given set of disease nodes.  
The drug repurposing inference script is executed from the command line using `inference_drug_repurposing.py` and produces ranked predictions with confidence scores.

#### Command
python inference_drug_repurposing.py [arguments]


#### Arguments

- *--gpu*  
  GPU device ID  
  **Type:** int  
  **Default:** `0`

- *--head_csv*  
  Path to a CSV file containing disease node indices  
  **Type:** string  
  **Required:** yes

- *--batch_size*  
  Batch size for inference.  
  Must be smaller than the number of provided disease nodes.  
  **Type:** int  
  **Default:** `1`

- *--N_top*  
  Number of top-ranked drug node predictions returned per disease node  
  **Type:** int  
  **Default:** `20`

#### Note

- The CSV file provided via *--head_csv* must contain valid node indices consistent with the knowledge graph.

### Multi-task inference

Multi-task inference enables prediction across various biomedical relations supported by the knowledge graph (e.g., indication, drug–target, drug–ADR, drug–disease, contraindications).  
The multi-task inference script is executed from the command line using `inference_multi_task.py` and produces ranked predictions for tail nodes.


#### Command

python inference_multi_task.py [arguments]


#### Arguments

- *--gpu*  
  GPU device ID  
  **Type:** int  
  **Default:** `0`

- *--head_csv*  
  Path to a CSV file containing head node indices  
  **Type:** string  
  **Required:** yes

- *--head_type*  
  Type of the head nodes  
  **Type:** string  
  **Default:** `drug`

- *--tail_type*  
  Type of the tail nodes to be predicted  
  **Type:** string  
  **Default:** `gene/protein`

- *--relation_type*  
  Relation type between head and tail nodes  
  **Type:** string  
  **Default:** `drug_protein`

- *--N_top*  
  Number of top-ranked tail node predictions returned per head node  
  **Type:** int  
  **Default:** `20`

- *--batch_size*  
  Batch size for inference.  
  Must be smaller than the number of provided head nodes.  
  **Type:** int  
  **Default:** `1`

#### Example
```
python inference_multi_task.py \
  --gpu 0 \
  --head_csv data/heads.csv \
  --head_type drug \
  --tail_type gene/protein \
  --relation_type drug_protein \
  --N_top 20 \
  --batch_size 1
```

#### Notes

- The values of *--head_type*, *--tail_type*, and *--relation_type* must be consistent with the knowledge graph.
- The CSV file provided via *--head_csv* must contain valid node indices.

---

## Subnetwork Generation

Subnetwork generation enables context-specific mechanistic interpretation by extracting informative subnetworks conditioned on a given query node and relation type.  
The generated subnetworks highlight biologically and pharmacologically relevant connections learned by NetMedGPT.
The subnetwork generation script is executed from the command line using `subnetwork_generator.py`. 


#### Command

python subnetwork_generator.py [arguments]


#### Arguments

- *--gpu*  
  GPU device ID  
  **Type:** int  
  **Default:** `0`

- *--head_index*  
  Node index of the head entity  
  **Type:** int or list of int  
  **Default:** `[14016]`

- *--head_type*  
  Type of the head node  
  **Type:** string  
  **Default:** `drug`

- *--tail_type*  
  Type of the tail nodes used to construct the subnetwork  
  **Type:** string  
  **Default:** `gene/protein`

- *--relation_type*  
  Relation type guiding subnetwork extraction  
  **Type:** string  
  **Default:** `drug_protein`

- *--N_top*  
  Number of top-ranked tail nodes used to build the subnetwork  
  **Type:** int  
  **Default:** `5`


#### Example
```
python generate_subnetwork.py \
  --gpu 0 \
  --head_index 14016 \
  --head_type drug \
  --tail_type gene/protein \
  --relation_type drug_protein \
  --N_top 20 \
  --batch_size 1
```

#### Notes
- The *--head_index* must correspond to valid node indices in the knowledge graph.
- The combination of *--head_type*, *--tail_type*, and *--relation_type* must be consistent with the knowledge graph.
- The output subnetwork can be used for downstream visualization and mechanistic analysis.









