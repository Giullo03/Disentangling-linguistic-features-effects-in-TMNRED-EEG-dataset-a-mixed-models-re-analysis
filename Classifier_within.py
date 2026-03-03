from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression #for binary classification
from sklearn.model_selection import StratifiedKFold #fro corss-validation
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle #for permutation test


base_path = Path(r"Path-to-dataset")
file_path = base_path / "dataset-name"
df = pd.read_excel(file_path)

df = df[df["repetition"] == 1].copy() #delete double sentences

# select features used for classifiers
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


#define cross-validation function within subject
def within_subject_cv(df, features):

    subjects = df["sub"].unique()
    subject_scores = [] #mean accuracy for each subject

    for sub in subjects:
        sub_df = df[df["sub"] == sub] #filtered df for current subject

        X = sub_df[features].values #df filtered for features
        y = sub_df["Group"].values  #0 = nontarget, 1 = target

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_scores = [] #accuracy for each fold

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),                               #normalize
                ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"))    #L2 beacuse of small number of features
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            score = balanced_accuracy_score(y_test, y_pred)
            fold_scores.append(score)

        subject_scores.append(np.mean(fold_scores))

    return np.array(subject_scores)

# compute mean score for each set of future across subjects
nlp_scores = within_subject_cv(df, nlp_features)
eeg_scores = within_subject_cv(df, eeg_features)
combined_scores = within_subject_cv(df, combined_features)

print("NLP mean accuracy:", np.mean(nlp_scores))
print("EEG mean accuracy:", np.mean(eeg_scores))
print("Combined mean accuracy:", np.mean(combined_scores))


# Permutation Test
def permutation_test(df, features, n_perm=1000):
    real_scores = within_subject_cv(df, features)
    real_mean = np.mean(real_scores)
    
    perm_means = []
    for i in tqdm(range(n_perm), desc=f"Permuting {features[0][:3]}..."):
        # within subject labels shuffling
        df_perm = df.copy()
        df_perm["Group"] = df_perm.groupby("sub")["Group"].transform(np.random.permutation)
        
        perm_scores = within_subject_cv(df_perm, features)
        perm_means.append(np.mean(perm_scores))
    
    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= real_mean)

    return real_mean, perm_means, p_value


real_nlp, perm_nlp, p_nlp = permutation_test(df, nlp_features)
real_eeg, perm_eeg, p_eeg = permutation_test(df, eeg_features)
real_comb, perm_comb, p_comb = permutation_test(df, combined_features)

# Output
results = [
    ("NLP", real_nlp, p_nlp),
    ("EEG", real_eeg, p_eeg),
    ("Combined", real_comb, p_comb)
]

print("CLASSIFICATION RESULTS")
for name, score, p in results:
    sig = "*" if p < 0.05 else "n.s."
    print(f"{name:10} | Accuracy: {score:.4f} | p-value: {p:.4f} [{sig}]")


