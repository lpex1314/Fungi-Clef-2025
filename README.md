# AMLS_II_assignment24_25/ ProtoNet: Few-Shot Fine-Grained Fungal Species Classification

This repository contains the official implementation of my solution to the **FungiCLEF 2025 Challenge**, a few-shot fine-grained visual classification competition hosted on [Kaggle](https://www.kaggle.com/competitions/fungi-clef-2025/data).

🏆 Final Result: 
📈 Public Top-5 Accuracy: **0.47787** (outperforming official BioCLIP + FAISS baseline)

---

## 🔍 Overview

This project leverages **BioCLIP** — a biodiversity-adapted CLIP model — and adapts it with a **Prototypical Network** architecture to handle long-tailed, few-shot classification across >2,000 fungal species.

Key components:

-  **FungiEmbedder**: A fine-tuned image encoder based on BioCLIP (ViT backbone), with selective unfreezing of top layers.
-  **Fixed Prototypes**: Class-wise mean embeddings are precomputed and frozen as class prototypes.
-  **Prototypical Loss**: Distance-based cross-entropy loss for pulling embeddings closer to their class prototype.
-  **Multi-image aggregation**: Average pooling across images from the same observation improves robustness.

---

## 📁 Folder Structure

```bash
.
├── main.py                  # Entry point: training, validation, inference
├── models.py                # FungiTastic dataset class, FungiEmbedder and Prototypical Network
├── statistics.ipynb         # Visualization for class distribution of training set
├── config.py                # Config file (hyperparameters etc.)
├── results/                 # Submission CSVs, checkpoints, logs
├── data/fungi-clef-2025     # Datasets
├── BioCLIP.txt              # Parameters name for BioCLIP model
├── training_curve.py        # Code for plotting
├── training_curve.png       # training curve
└── README.md                # You are here
```

## 🚀 Quick Start
Download the FungiCLEF 2025 dataset from Kaggle, unzip and save to `data/`.

Install dependencies (requires Python 3.9+ and CUDA-enabled GPU):
```
pip install -r requirements.txt
```

Run training and Generate submission:
```
python main.py
```
## 📊 Experimental Results

Setting	Top-5 Accuracy \
BioCLIP + FAISS + Prototypes (baseline)	0.33185 \
Max pooling (resblocks.10–11)	0.42477 \
Average pooling (resblocks.10–11)	0.44247 \
Average pooling + more fine-tuned layers	0.47787 

## 🧠 Key Insight
While Prototypical Networks with frozen prototypes offer strong few-shot generalization, fixing the prototype positions limits class separability. Future work could explore learnable prototypes, margin-based loss, and dynamic prototype updates to further optimize inter-class distances.

