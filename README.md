# A Mixed-Effects re-analysis of TMNRED EEG dataset: disentangling linguistic features effects 
This repository contains the code and analyses for the study "A Mixed-Effects re-analysis of TMNRED EEG dataset: disentangling linguistic features effects", which investigated the relationships between EEG features, linguistic features and semantic processing. 

The full manuscript is available in:
- `Paper.pdf`

--- 

## Dataset
The EEG dataset used in this project is publicly available at:
https://openneuro.org/datasets/ds005383

In order to process the dataset by using the pipelines avalable in this repository you will need to mintain the original BIDS structure.

--- 
## Project Overview
The Pipeline consists fo four main stages:

1. EEG preprocessing and feature extraction (MNE)
2. NLP feature extraction (Stanza, GPT2, SBERT)
3. Classification 
4. Representational Similarity Analysis (RSA)

---
# 1.EEG Preprocessing andfeature extraction
**File: `MNE_TMNRED.py`**

For each subject and session:

- Bad channel detection
- Interpolation
- Re-referencing
- Filtering (0.5-80 bandpass, 50Hz notch)
- Downsampling (200Hz)
- ICA
- Epoching
- Feature extraction: mean amplitude per channel early (0-0.5s) and late(0.5-2s); Frequency bands (theta, alpha, beta, gamma)
- Quality Check report for each subject for each session: number and % of bad channels; number and % of rejected epochs

---
# 2. NLP feature extraction
**File: `NLP_TMNRED.py`**
## Model used:
- Stanza: chinese pipeline with tokenize, POS,lemma, depparse
- GPT2: uer/gpt2-cinese-cluecorpussmall
- Sentence-BERT: paraphrase-multilingual-MiniLM-L12-v2

## Extracted NLP features:
For each sentence:
- Syntactic complexity
- Mean Surprisal
- Mean stroke count
- Mean haracter frequency
- Mean word frequency
- Cosine similarity matrix

---

# 3. Classification
**File: `Classifier_within.py`**
- 5-fold cross-validation
- logistic regression 
- blanced accuracy
- permutation test (1000 permutations)

**`Classifier_across.py`**
- leave-one-group-out
- logistic regression 
- balanced accuracy
- permutation test (1000 permutations)

---
# 4.Rappresentational Similarity Analysis (RSA)
- **File: `RSA.py`** 
- compute EEG rappresentational Dissimilarity Matrices (RDM)
- convert semantic RSM to RDM 
- spearman correlation between upper triangles
- permutation test (sign-flipping, 10000 iterations)
- cohen's d
---

# Requirments
**File: `requirments.txt`**

---

# Author: Sesini Giulio
