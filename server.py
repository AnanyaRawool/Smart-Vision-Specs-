import base64
import torch
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify
from PIL import Image

# Initialize Flask app
app = Flask(name)

# Load YOLOv5 model
print("🔍 Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')

# Load the EAST text detection model
EAST_MODEL = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(EAST_MODEL)

# Set up Tesseract OCR path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def decode_image(base64_string):
    """Decode base64 image to OpenCV format."""
    try:
        img_data = base64.b64decode(base64_string)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠ Error decoding image: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return processed

def detect_text_regions(image):
    """Detect text regions using EAST text detection model."""
    orig = image.copy()
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    
    image = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    output = net.forward(layerNames)
    
    if output is None or len(output) != 2:
        return []
    
    scores, geometry = output
    rectangles, confidences = [], []

    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            score = scores[0, 0, y, x]
            if score < 0.5:
                continue
            angle = geometry[0, 4, y, x]
            cosA, sinA = np.cos(angle), np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            offsetX, offsetY = x * 4.0, y * 4.0
            endX, endY = int(offsetX + (cosA * w) + (sinA * h)), int(offsetY + (sinA * w) - (cosA * h))
            startX, startY = int(endX - w), int(endY - h)
            rectangles.append((startX, startY, endX, endY))
            confidences.append(float(score))
    
    indices = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.4)
    if len(indices) == 0:
        return []
    if isinstance(indices, tuple):
        indices = indices[0]
    
    return [rectangles[i] for i in indices.flatten()]

def extract_text(image):
    """Perform OCR on detected text regions."""
    processed = preprocess_image(image)
    boxes = detect_text_regions(image)
    extracted_text = ""

    for (x1, y1, x2, y2) in boxes:
        roi = processed[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.size == 0:
            continue
        text = pytesseract.image_to_string(roi, config="--psm 6")
        extracted_text += text.strip() + " "

    return extracted_text.strip()

@app.route('/process_image', methods=['POST'])
def process_data():
    """Flask endpoint for processing image: Object Detection + OCR."""
    try:
        print("📥 Received POST request...")
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        frame = decode_image(data["image"])
        if frame is None:
            return jsonify({"error": "Image decoding failed"}), 400

        print("🔍 Running YOLOv5 detection...")
        results = model(frame)
        labels = []

        if results is not None and len(results.xyxy[0]) > 0:
            labels = list(set(model.names[int(det[-1])] for det in results.xyxy[0]))  # Fixed YOLOv5 issue

        print("📖 Running OCR for text detection...")
        extracted_text = extract_text(frame)

        return jsonify({"objects": labels, "text": extracted_text})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if name == 'main':
    app.run(host='0.0.0.0', port=5000, debug=True)
