# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime

# Initialize Flask
app = Flask(__name__)

MODEL_PATH = "malaria_model.h5"

print("\nLoading malaria_model.h5 ...")

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print("Error loading model:", e)
    model = None

# Image Preprocessing
def prepare_image(image_file):
    try:
        img = Image.open(image_file)
        img = img.convert("RGB")          # Ensure 3 channels
        img = img.resize((64, 64))       # SAME SIZE as training
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Image processing error:", e)
        return None

# Home Route with Upload Form
@app.route('/', methods=['GET', 'POST'])
def home():
    result_html = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            result_html = "<p style='color:red;'>No file uploaded!</p>"
        else:
            file = request.files['file']
            if file.filename == '':
                result_html = "<p style='color:red;'>No file selected!</p>"
            else:
                img_array = prepare_image(file)
                if img_array is None:
                    result_html = "<p style='color:red;'>Invalid image!</p>"
                else:
                    pred = model.predict(img_array, verbose=0)[0][0]
                    if pred > 0.5:
                        label = "Uninfected"
                        confidence = pred * 100
                    else:
                        label = "Parasitized"
                        confidence = (1 - pred) * 100
                    result_html = f"""
                    <p>Prediction: <b>{label}</b></p>
                    <p>Confidence: <b>{confidence:.2f}%</b></p>
                    <p>Raw Probability: <b>{pred:.4f}</b></p>
                    <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """

    # Simple HTML form template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Malaria Cell Detection</title>
        <style>
            body {{font-family:Arial; background:#f0f2f5; text-align:center; padding:50px;}}
            .container {{background:white; padding:40px; border-radius:10px; display:inline-block;}}
            input[type=file] {{margin:20px 0;}}
            button {{padding:10px 20px; font-size:16px;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦟 Malaria Cell Detection</h1>
            <h3>Model Status: {'LOADED' if model else 'NOT LOADED'}</h3>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Upload & Predict</button>
            </form>
            <div style="margin-top:30px;">
                {result_html}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

# API Route (Optional, still works)
@app.route('/predict', methods=['POST'])
def predict_api():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    img_array = prepare_image(file)
    if img_array is None:
        return jsonify({'success': False, 'error': 'Invalid image'}), 400

    pred = model.predict(img_array, verbose=0)[0][0]

    if pred > 0.5:
        label = "Uninfected"
        confidence = pred * 100
    else:
        label = "Parasitized"
        confidence = (1 - pred) * 100

    return jsonify({
        'success': True,
        'prediction': label,
        'confidence': round(confidence, 2),
        'raw_probability': float(pred),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


# Run App
if __name__ == '__main__':
    print("\nServer running at: http://127.0.0.1:5000\n")
    app.run(debug=True)
