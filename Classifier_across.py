from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

# Load
base_path = Path(r"Path-to-dataset")
file_path = base_path / "dataset name"
df = pd.read_excel(file_path)

df = df[df["repetition"] == 1].copy()

# Features
nlp_features = [
    "Z_SyntacticComplexity",
    "Z_Surprisal",
    "Z_Strokes",
    "Z_char_frequency",
    "Z_word_frequency"
]

eeg_features = [
    "Z_Late_Cz",
    "Z_Late_C3",
    "Z_Alpha_Early_Global",
    "Z_beta_early_left"
]

combined_features = nlp_features + eeg_features

# Across Subject  CV
def across_subject_cv(df, features):

    X = df[features].values
    y = df["Group"].values
    groups = df["sub"].values

    logo = LeaveOneGroupOut()
    subject_scores = []

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        score = balanced_accuracy_score(y_test, y_pred)
        subject_scores.append(score)

    return np.array(subject_scores)

# Permutation test
def permutation_test_across(df, features, n_perm=1000):

    real_scores = across_subject_cv(df, features)
    real_mean = np.mean(real_scores)

    perm_means = []

    for i in tqdm(range(n_perm), desc=f"Permuting {features[0][:3]}..."):

        df_perm = df.copy()

        # shuffle labels WITHIN each subject
        df_perm["Group"] = (
            df_perm.groupby("sub")["Group"]
            .transform(np.random.permutation)
        )

        perm_scores = across_subject_cv(df_perm, features)
        perm_means.append(np.mean(perm_scores))

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= real_mean)

    return real_mean, perm_means, p_value

# Compute
real_nlp, perm_nlp, p_nlp = permutation_test_across(df, nlp_features)
real_eeg, perm_eeg, p_eeg = permutation_test_across(df, eeg_features)
real_comb, perm_comb, p_comb = permutation_test_across(df, combined_features)

# output
results = [
    ("NLP", real_nlp, p_nlp),
    ("EEG", real_eeg, p_eeg),
    ("Combined", real_comb, p_comb)
]

print("ACROSS-SUBJECT CLASSIFICATION")
for name, score, p in results:
    sig = "*" if p < 0.05 else "n.s."
    print(f"{name:10} | Accuracy: {score:.4f} | p-value: {p:.4f} [{sig}]")

