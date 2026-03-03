from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr


eeg_path = Path(r"eeg-dataset-path")
semantic_path = Path(r"similarity-matrix-path")

eeg = pd.read_excel(eeg_path)
semantic = pd.read_excel(semantic_path, index_col=0)

print("EEG shape:", eeg.shape)
print("Semantic shape:", semantic.shape)

eeg_features = [
    "Z_Late_Cz",
    "Z_Late_C3",
    "Z_Late_Pz",
    "Z_Alpha_Late_Global",
    "Z_beta_late_left"
]

unique_sentences = eeg["label_doubles"].unique()

semantic_reduced = semantic.loc[unique_sentences, unique_sentences]


def compute_rdm(df, features):
    X = df[features].values
    return squareform(pdist(X, metric="correlation"))

# test whether there is stability between RDM computed using only the first occurence of the sentences that repeats across blocks 
# and the one computed using the only second occurence
stability_scores = []
subjects = eeg["sub"].unique()

for s in subjects:
    
    sub = eeg[eeg["sub"] == s]
    
    rep1 = sub[sub["repetition"] == 1]
    rep2 = sub[sub["repetition"] == 2]
    
    rep1_avg = rep1.groupby("label_doubles")[eeg_features].mean()
    rep2_avg = rep2.groupby("label_doubles")[eeg_features].mean()
    
    common = rep1_avg.index.intersection(rep2_avg.index)
    
    if len(common) < 10:
        continue
    
    rdm1 = compute_rdm(rep1_avg.loc[common], eeg_features)
    rdm2 = compute_rdm(rep2_avg.loc[common], eeg_features)
    
    triu = np.triu_indices(len(common), 1)
    
    corr, _ = spearmanr(rdm1[triu], rdm2[triu])
    stability_scores.append(corr)

print("Mean rho between rep1-rep2", np.mean(stability_scores))  # there is no stability, therefore instead of averaging across repetitions the second occurences will be delted

# Narrow down the eeg matrix and the semantic similarity matrix
eeg_rep1 = eeg[eeg["repetition"] == 1].copy()

unique_sentences = eeg_rep1["label_doubles"].unique()

semantic_reduced = semantic.loc[unique_sentences, unique_sentences]

# RSA subject-level
rsa_scores = []
subjects = eeg_rep1["sub"].unique()

for s in subjects:
    sub = eeg_rep1[eeg_rep1["sub"] == s]
    sub_avg = sub.groupby("label_doubles")[eeg_features].mean()
    sentences_present = sub_avg.index
   
    eeg_rdm = compute_rdm(sub_avg, eeg_features)
    
    semantic_sub = semantic_reduced.loc[sub_avg.index, sub_avg.index]
    assert all(sub_avg.index == semantic_sub.index)

    semantic_rdm = 1 - semantic_sub.values

    triu = np.triu_indices(len(sentences_present), 1)
    
    corr, _ = spearmanr(eeg_rdm[triu], semantic_rdm[triu])
    
    rsa_scores.append(corr)

rsa_scores = np.array(rsa_scores)
mean_rsa = rsa_scores.mean()

n_perm = 10000
perm_means = []
for i in range(n_perm):
    flipped = rsa_scores * np.random.choice([-1, 1], size=len(rsa_scores))
    perm_means.append(flipped.mean())
perm_means = np.array(perm_means)
p_perm = np.mean(np.abs(perm_means) >= np.abs(mean_rsa))

d = mean_rsa / rsa_scores.std(ddof=1)


print("Final mean rho", mean_rsa)
print("Permutation p-value:", p_perm)
print("Cohen's d:", d)

# Plot heatmaps
semantic_rdm_group = 1 - semantic_reduced.values
plt.figure()
plt.imshow(semantic_rdm_group)
plt.title("Semantic RDM (rep1)")
plt.colorbar()
plt.show()


group_avg = (
    eeg_rep1
    .groupby("label_doubles")[eeg_features]
    .mean()
)
group_avg = group_avg.loc[semantic_reduced.index]
eeg_rdm_group = squareform(pdist(group_avg.values, metric="correlation"))
plt.figure()
plt.imshow(eeg_rdm_group)
plt.title("EEG RDM (group average, rep1)")
plt.colorbar()
plt.show()


