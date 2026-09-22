
#  Transformer From Scratch: "Attention Is All You Need"

An educational PyTorch implementation of the classic **"Attention Is All You Need"** paper (Vaswani et al., 2017).  
This project builds the Transformer architecture **from scratch** — covering data preprocessing, tokenization, model definition, training loop, and validation — without relying on high-level libraries like `transformers`.

---

## ✏️ **Project Overview**
This implementation includes:
-  Custom Transformer encoder–decoder
-  Tokenization using `tokenizers` library (WordLevel)
-  Dataset preparation for bilingual translation (using `opus_books` dataset)
-  Training loop with cross-entropy loss & label smoothing
-  Greedy decoding for inference / validation
-  Logging with TensorBoard
-  Model checkpointing

---

---

## 📦 **Installation**
1. Clone the repository:
```bash
git clone https://github.com/GhaziahShoeb/transformer.git
cd transformer/"transformer from scratch"

python train.py
```
