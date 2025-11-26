#!/usr/bin/env python3
import joblib

# Path to your saved model
MODEL_PATH = "./model_output/payload_classifier_tfidf_lr.joblib"

# Load the model
model = joblib.load(MODEL_PATH)
print("Model loaded!")

# Example payloads to test
payloads = [
    "alert('XSS')",              # XSS example
    "1 OR 1=1",                  # SQLi example
    "${7*7}",                     # SSTI example
    "normal text",               # benign
    "`rm -rf /`"                 # CMDi example
]

# Predict labels
preds = model.predict(payloads)

# Optionally, get probabilities
probs = model.predict_proba(payloads)

for text, label, prob in zip(payloads, preds, probs):
    print(f"Payload: {text}\nPredicted label: {label}\nProbabilities: {prob}\n")
