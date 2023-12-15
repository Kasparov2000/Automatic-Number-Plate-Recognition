import streamlit as st
from skimage import io
import cv2
import base64
import pytesseract as pt
import numpy as np
from io import BytesIO
import secrets
import os

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Global variable for the result image
result_image = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        return text


def is_license_plate(confidence, class_score):
    # Define the confidence and class_score thresholds
    confidence_threshold = 0.4
    class_score_threshold = 0.25

    return confidence > confidence_threshold and class_score > class_score_threshold


def get_detections(img, net):
    # 1. CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_suppression(input_image, detections):
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILITY SCORE
    # center x, center y, w, h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        class_score = row[5]  # probability score of license plate

        if is_license_plate(confidence, class_score):
            cx, cy, w, h = row[0:4]

            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])

            confidences.append(confidence)
            boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index


def drawings(image, boxes_np, confidences_np, index):
    license_text = ""
    # 5. Drawings
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image, license_text


def yolo_predictions(img, net):
    global result_image

    # step-1: detections
    input_image, detections = get_detections(img, net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    # step-3: Drawings
    result_img, license_text = drawings(img, boxes_np, confidences_np, index)

    # Save the result image
    result_image = result_img

    return license_text


def main():
    st.title("AUTOMATIC NUMBER-PLATE RECOGNITION")

    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'gif'])

    if uploaded_file is not None:
        # Read the image from BytesIO
        img = io.imread(uploaded_file)

        filename = secrets.token_hex(8)
        license_text = yolo_predictions(img, net)

        if 'irrelevant' in license_text.lower():
            st.image(result_image, caption="License Plate Image", use_column_width=True)
            st.error("Irrelevant Object Detected")
        else:
            st.image(result_image, caption="License Plate Image", use_column_width=True)
            st.success(f"License Plate Number: {license_text}")


if __name__ == '__main__':
    main()
