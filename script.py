#!/usr/bin/env python3
"""
train_payload_classifier.py

Train a payload classifier (benign / xss / sqli / ssti / cmdi / other)
using TF-IDF char n-grams + LogisticRegression with hyperparameter tuning.

Requirements:
  pip install scikit-learn pandas joblib umap-learn --user
(umap-learn not required; it's only if you want visualization later)

Run:
  python train_payload_classifier.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# -------- CONFIG --------
DATA_PATH = "./payload_dataset.json"   # change if needed
OUT_DIR = "./model_output"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# -------- LOAD DATA --------
print("Loading dataset from", DATA_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
# Some dataset versions use key 'input' or 'text' — normalize:
if "input" in df.columns and "text" not in df.columns:
    df = df.rename(columns={"input": "text"})
if "text" not in df.columns:
    raise ValueError("Dataset must have 'text' or 'input' field. Found columns: " + ", ".join(df.columns))

# Drop any empty text rows and reset
df = df[df["text"].map(lambda x: isinstance(x, str) and x.strip() != "")]
df = df.reset_index(drop=True)

print("Total samples:", len(df))
print("Class distribution:\n", df["label"].value_counts())

# -------- TRAIN / TEST SPLIT (stratified) --------
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train size:", len(X_train), "Test size:", len(X_test))

# -------- CLASS WEIGHTS (optional) --------
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {c: w for c, w in zip(classes, cw)}
print("Computed class weights:", class_weights)

# -------- PIPELINE (TF-IDF char n-grams -> LogisticRegression) --------
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),      # good for payload patterns (symbols + small tokens)
    max_features=75000,      # limit features to control memory; tune if needed
    sublinear_tf=True,
)

lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    max_iter=2000,
    class_weight=class_weights,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

pipe = Pipeline([
    ("tfidf", tfidf),
    ("clf", lr)
])

# -------- HYPERPARAMETER GRID (small but effective) --------
param_grid = {
    "tfidf__ngram_range": [(2,4), (2,5), (2,6)],
    "tfidf__max_features": [30000, 50000, 75000],
    "clf__C": [0.1, 1.0, 5.0]
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

# -------- TRAINING --------
print("Starting grid search training...")
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score (f1_macro):", grid.best_score_)

# Save best estimator
best_model = grid.best_estimator_
joblib.dump(best_model, os.path.join(OUT_DIR, "payload_classifier_tfidf_lr.joblib"))
print("Saved best model to", os.path.join(OUT_DIR, "payload_classifier_tfidf_lr.joblib"))

# Also save grid object (for later analysis)
joblib.dump(grid, os.path.join(OUT_DIR, "grid_search_payload.joblib"))

# -------- EVALUATION ON TEST SET --------
print("Evaluating on test set...")
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy: %.4f" % acc)

report = classification_report(y_test, y_pred, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))
print("Saved classification report to classification_report.csv")
print(report_df)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.named_steps["clf"].classes_)
cm_df = pd.DataFrame(cm, index=best_model.named_steps["clf"].classes_, columns=best_model.named_steps["clf"].classes_)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))
print("Saved confusion matrix to confusion_matrix.csv")

# Save test predictions for inspection
preds_out = pd.DataFrame({"text": X_test, "label_true": y_test, "label_pred": y_pred})
preds_out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
print("Saved test predictions to test_predictions.csv")

print("All done. Models and reports are under:", OUT_DIR)
