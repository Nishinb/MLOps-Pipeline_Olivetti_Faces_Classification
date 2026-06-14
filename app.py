import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
MODEL_PATH = "savedmodel.pth"

# Minimal UI template embedded directly for simple containerization
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>MLOps Assignment: Face Classifier</title></head>
<body>
    <h2>Olivetti Face Classifier Engine</h2>
    <p>Submit 4096 comma-separated pixel values (64x64 flattened normalized image):</p>
    <form action="/predict" method="post">
        <textarea name="pixels" rows="10" cols="50" placeholder="0.5, 0.2, 0.1..."></textarea><br><br>
        <input type="submit" value="Classify Face Profile">
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model artifact file missing"}), 500
            
        raw_data = request.form.get('pixels') or request.json.get('pixels')
        if not raw_data:
            return jsonify({"error": "No pixel input payload provided"}), 400
            
        # Parse inputs
        pixels = np.fromstring(raw_data, sep=',', dtype=np.float32)
        if len(pixels) != 4096:
            return jsonify({"error": f"Invalid dimension. Expected 4096 values, got {len(pixels)}"}), 400
            
        model = joblib.load(MODEL_PATH)
        prediction = int(model.predict([pixels])[0])
        return jsonify({"status": "success", "predicted_class": prediction})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)