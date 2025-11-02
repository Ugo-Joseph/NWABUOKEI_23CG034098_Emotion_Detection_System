from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import os
from datetime import datetime

app = Flask(__name__)

# ✅ Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "creative_model_name.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Please train it first using model.py")

model = load_model(MODEL_PATH)

# ✅ Class labels (same as your training data)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# ✅ Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face_img, target_size=(48, 48)):
    """Resize, normalize, and reshape the face for CNN model input."""
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, target_size)
    face = face / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face


@app.route('/')
def index():
    """Homepage."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle uploaded image or webcam snapshot and return prediction."""
    img = None
    name = request.form.get("name", "Anonymous").strip()

    # --- Handle uploaded image ---
    if 'file' in request.files:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'})
        img = Image.open(file).convert('RGB')

    # --- Handle webcam image (base64 JSON) ---
    elif request.is_json and 'webcam' in request.json:
        img_data = request.json['webcam']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # --- If no image provided ---
    if img is None:
        return jsonify({'error': 'No image provided'})

    # --- Convert PIL image to OpenCV format ---
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # --- Resize if image is too large ---
    max_width = 800
    if img_bgr.shape[1] > max_width:
        scale = max_width / img_bgr.shape[1]
        img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)

    # --- Detect face(s) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Try better lighting or move closer.'})

    # --- Pick the largest detected face ---
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_bgr = img_bgr[y:y + h, x:x + w]

    # --- Preprocess and predict ---
    face_input = preprocess_face(face_bgr)
    preds = model.predict(face_input)
    emotion = class_labels[int(np.argmax(preds))]

    # --- Annotate image ---
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img_bgr, emotion.capitalize(), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    # --- Convert to base64 for web display ---
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'emotion': emotion, 'image': img_b64})


if __name__ == '__main__':
    app.run(debug=True)
