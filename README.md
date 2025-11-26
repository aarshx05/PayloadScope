# PayloadScope: Intelligent Malicious Payload Classification System

PayloadScope is a production-focused machine learning system designed to classify potentially malicious payloads used in web exploitation attempts. The solution supports both a **Flask-based REST API** and a **standalone GUI application**, making it suitable for automated pipelines as well as manual analysis workflows.

The classifier identifies multiple categories of attack vectors, including XSS, SQL Injection, Server-Side Template Injection, Command Injection, and benign inputs. It includes a robust decoding engine capable of handling various single and multi-layer encoding schemes to normalize payloads before analysis.

---

## Features

### Machine Learning Pipeline

* TF-IDF character-level n-gram vectorization.
* Logistic Regression multi-class classifier.
* Confidence-based scoring and threshold configuration.
* Encoding normalization for URL, Base64, HTML entities, and double-encoded payloads.

### Dual Operation Modes

**1. Flask API**

* REST endpoint for real-time classification.
* Low-latency inference design.
* Suitable for CI/CD, WAF evaluation, SIEM/SOAR integration, and automated triage.

**2. GUI Application**

* Desktop interface for analysts and researchers.
* Single-input inspection workflow.
* Displays decoded payloads, predicted class, and confidence score.

### Additional Advantages

* Modular training and inference architecture.
* Clean separation between model lifecycle and serving layer.
* Easily extendable to new payload categories or pre-processing logic.

---

## Project Structure

```
PayloadScope/
│
├── model_training/
│   ├── train_payload_classifier.py
│   ├── preprocess.py
│   └── model_output/
│       └── payload_classifier_tfidf_lr.joblib
│
├── api/
│   ├── app.py
│   └── utils.py
│
├── gui/
│   └── payload_gui.py
│
├── tests/
│   └── sample_payloads.json
│
└── README.md
```

---

## Setup Instructions

### 1. Create and Activate Virtual Environment

```
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running the Flask API

### Start the Server

```
cd api
python app.py
```

### Example Request

```
POST /classify
{
    "payload": "<script>alert(1)</script>"
}
```

### Example Response

```json
{
    "normalized": "<script>alert(1)</script>",
    "predicted_class": "xss",
    "confidence": 0.94
}
```

---

## Running the GUI Application

```
cd gui
python payload_gui.py
```

The interface allows entering payloads manually, reviewing decoded values, and viewing model predictions with confidence scoring.

---

## Training the Model

To retrain or experiment with improved datasets:

```
cd model_training
python train_payload_classifier.py
```

The resulting model file is saved under:

```
model_output/payload_classifier_tfidf_lr.joblib
```

---

## Use Cases

* Security research and payload behavioral analysis
* Automated triage in bug bounty or internal security operations
* WAF evaluation and signature testing
* CI/CD pipeline integration for secure development workflows
* Educational usage within controlled cybersecurity labs

---
