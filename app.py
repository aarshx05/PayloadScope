from flask import Flask, render_template, request, jsonify
import joblib
import base64
import binascii
import urllib.parse
import re

app = Flask(__name__)

# Load model
MODEL_PATH = "./model_output/payload_classifier_tfidf_lr.joblib"
model = joblib.load(MODEL_PATH)
classes = model.named_steps["clf"].classes_

# Threshold for confident predictions
THRESHOLD = 0.80  # 80%

def try_base64_decode(s):
    """Try Base64 decode. Return original if fails."""
    try:
        s_bytes = s.encode()
        missing_padding = len(s_bytes) % 4
        if missing_padding:
            s_bytes += b'=' * (4 - missing_padding)
        decoded = base64.b64decode(s_bytes, validate=True)
        return decoded.decode('utf-8', errors='ignore')
    except (binascii.Error, UnicodeDecodeError):
        return s

def try_url_decode(s):
    """Try URL decode."""
    try:
        decoded = urllib.parse.unquote(s)
        return decoded
    except Exception:
        return s

def try_hex_decode(s):
    """Try hex decode if it looks like hex."""
    try:
        # Remove possible 0x or \x
        s_clean = re.sub(r'(\\x|0x)', '', s)
        # Must be even length
        if len(s_clean) % 2 != 0:
            return s
        decoded = bytes.fromhex(s_clean)
        return decoded.decode('utf-8', errors='ignore')
    except Exception:
        return s

def recursive_decode(payload, max_depth=2):
    """Try multiple decodings up to max_depth layers."""
    current = payload
    for _ in range(max_depth):
        prev = current
        # Try Base64
        current = try_base64_decode(current)
        # Try URL decode
        current = try_url_decode(current)
        # Try Hex decode
        current = try_hex_decode(current)
        # Stop if no change
        if current == prev:
            break
    return current

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    payload = data.get("payload", "")

    if not payload.strip():
        return jsonify({"error": "No payload provided"}), 400

    decoded_payload = recursive_decode(payload, max_depth=2)

    # Prediction
    prob = model.predict_proba([decoded_payload])[0]
    pred_index = prob.argmax()
    pred_label = classes[pred_index]
    pred_conf = prob[pred_index]

    # Apply threshold
    if pred_conf < THRESHOLD:
        pred_label = "uncertain / likely benign"

    prob_dict = {cls: float(p) for cls, p in zip(classes, prob)}

    return jsonify({
        "original_payload": payload,
        "decoded_payload": decoded_payload,
        "prediction": pred_label,
        "confidence": float(pred_conf),
        "probabilities": prob_dict
    })

if __name__ == "__main__":
    app.run(debug=True)
