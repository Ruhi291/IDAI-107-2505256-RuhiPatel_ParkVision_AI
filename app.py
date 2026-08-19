"""
ParkVision AI - app.py (grid-based version, no extra dependencies)
User uploads a parking lot photo and specifies rows/columns of slots.
The app divides the image into that grid, classifies each cell as
occupied/empty, and displays results with metrics and recommendations.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from insight_logic import get_insights

st.set_page_config(page_title="ParkVision AI", layout="wide")

MODEL_PATH = "models/parkvision_model.h5"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def classify_slot(model, crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalized, axis=0)
    pred = model.predict(batch, verbose=0)[0][0]
    is_occupied = pred > 0.5
    return is_occupied, float(pred)


st.title("🅿️ ParkVision AI")
st.write(
    "Upload a parking lot photo and tell the app how many rows and columns "
    "of parking slots are visible. It will divide the image into that grid "
    "and classify each slot as occupied or empty."
)

uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

col_a, col_b = st.columns(2)
with col_a:
    rows = st.number_input("Number of rows of slots", min_value=1, max_value=20, value=3)
with col_b:
    cols = st.number_input("Number of columns of slots", min_value=1, max_value=20, value=6)

analyze_clicked = st.button("Analyze Parking Slots", type="primary")

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    if analyze_clicked:
        model = load_model()
        orig_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = orig_cv.shape[:2]
        annotated = orig_cv.copy()

        cell_w = w // cols
        cell_h = h // rows

        total_slots = 0
        occupied_slots = 0

        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = w if c == cols - 1 else (c + 1) * cell_w
                y2 = h if r == rows - 1 else (r + 1) * cell_h

                crop = orig_cv[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                is_occupied, confidence = classify_slot(model, crop)
                total_slots += 1

                if is_occupied:
                    occupied_slots += 1
                    color = (0, 0, 255)
                    label = f"{confidence:.0%}"
                else:
                    color = (0, 200, 0)
                    label = f"{1 - confidence:.0%}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.subheader("Annotated Result")
            st.image(annotated_rgb, use_container_width=True)
            st.caption("🟩 Green = Empty   🟥 Red = Occupied")

        insights = get_insights(total_slots, occupied_slots)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Slots", insights["total"])
        m2.metric("Occupied", insights["occupied"])
        m3.metric("Available", insights["available"])
        m4.metric("Occupancy %", f"{insights['occupancy_pct']}%")

        st.subheader(f"Congestion Level: {insights['congestion']}")

        if insights["congestion"] == "High":
            st.warning(insights["recommendation"])
        elif insights["congestion"] == "Low":
            st.success(insights["recommendation"])
        else:
            st.info(insights["recommendation"])
else:
    st.info("👆 Upload an image and set rows/columns to get started.")
