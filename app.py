"""
ParkVision AI - app.py
Intelligent Urban Parking Analytics & Space Optimisation Platform

Features:
- Grid-based slot occupancy detection (MobileNetV2 classifier)
- Recommendation logic (proceed / try another location)
- Model performance dashboard (accuracy, precision, recall, confusion matrix)
- Dataset info tab (class balance, preprocessing summary)
- Multi-lot management (named lots, comparison across lots)
- Session history with occupancy trend chart
- Occupancy heatmap across repeated analyses of the same lot layout
- Explainable AI: Grad-CAM overlay for a selected slot
- Video / simulated live-feed analysis (frame-by-frame occupancy trend)
- Rule-based Q&A assistant answering questions about the latest result
- One-click PDF report export
"""

import os
import sys
import io
import json
import datetime

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from fpdf import FPDF

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from insight_logic import get_insights

# ============================================================
# PAGE CONFIG & CONSTANTS
# ============================================================
st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")

# ============================================================
# THEME STATE & CUSTOM STYLING (fonts, colors, cards, header banner)
# ============================================================
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"


def get_custom_css(theme_mode):
    if theme_mode == "Light (Blue)":
        bg_grad = "linear-gradient(180deg, #eaf4ff 0%, #dceeff 100%)"
        sidebar_bg = "#dceeff"
        sidebar_border = "#b6d8f7"
        sidebar_text = "#0f2942"
        heading_color = "#0f2942"
        hero_grad = "linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%)"
        hero_h1 = "#ffffff"
        hero_p = "#eaf4ff"
        card_bg = "#ffffff"
        card_border = "#c7e0f8"
        metric_label = "#3b6a91"
        metric_value = "#1d4ed8"
        button_grad = "linear-gradient(90deg, #3b82f6, #2563eb)"
        button_shadow = "rgba(37,99,235,0.30)"
        tab_color = "#5c7fa3"
        tab_active = "#1d4ed8"
        uploader_bg = "#ffffff"
        uploader_border = "#93c5fd"
        hr_color = "#c7e0f8"
    else:
        bg_grad = "linear-gradient(180deg, #0b1120 0%, #0f172a 100%)"
        sidebar_bg = "#0b1120"
        sidebar_border = "#1e293b"
        sidebar_text = "#e2e8f0"
        heading_color = "#f1f5f9"
        hero_grad = "linear-gradient(135deg, #164e63 0%, #0f766e 50%, #065f46 100%)"
        hero_h1 = "#ffffff"
        hero_p = "#d1fae5"
        card_bg = "#111827"
        card_border = "#1f2937"
        metric_label = "#94a3b8"
        metric_value = "#34d399"
        button_grad = "linear-gradient(90deg, #0d9488, #059669)"
        button_shadow = "rgba(5,150,105,0.35)"
        tab_color = "#94a3b8"
        tab_active = "#34d399"
        uploader_bg = "#111827"
        uploader_border = "#334155"
        hr_color = "#1e293b"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {bg_grad};
}}

section[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    border-right: 1px solid {sidebar_border};
}}
section[data-testid="stSidebar"] * {{
    color: {sidebar_text} !important;
}}

h1, h2, h3 {{
    font-family: 'Poppins', sans-serif !important;
    color: {heading_color} !important;
}}

.pv-hero {{
    background: {hero_grad};
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}}
.pv-hero h1 {{
    color: {hero_h1} !important;
    font-size: 2rem;
    margin-bottom: 4px;
}}
.pv-hero p {{
    color: {hero_p};
    font-size: 1rem;
    margin: 0;
}}

div[data-testid="stMetric"] {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 16px 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}}
div[data-testid="stMetricLabel"] {{
    color: {metric_label} !important;
}}
div[data-testid="stMetricValue"] {{
    color: {metric_value} !important;
    font-family: 'Poppins', sans-serif !important;
}}

.stButton > button, .stDownloadButton > button {{
    background: {button_grad};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 14px {button_shadow};
    color: white;
}}

button[data-baseweb="tab"] {{
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: {tab_color};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {tab_active} !important;
    border-bottom: 3px solid {tab_active} !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 10px;
}}

section[data-testid="stFileUploaderDropzone"] {{
    background: {uploader_bg};
    border: 1px dashed {uploader_border};
    border-radius: 12px;
}}

hr {{
    border-color: {hr_color} !important;
}}
</style>
"""


st.markdown(get_custom_css(st.session_state["theme_mode"]), unsafe_allow_html=True)


def render_header(title, subtitle):
    st.markdown(f"""
    <div class="pv-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


MODEL_PATH = "models/parkvision_model.h5"
IMG_SIZE = (224, 224)
RESULTS_DIR = "results"
DATASET_DIR = "dataset"          # expects dataset/occupied and dataset/empty
HISTORY_CSV = os.path.join(RESULTS_DIR, "session_history.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

if "latest_insights" not in st.session_state:
    st.session_state["latest_insights"] = None
if "latest_lot_name" not in st.session_state:
    st.session_state["latest_lot_name"] = None
if "latest_crops" not in st.session_state:
    st.session_state["latest_crops"] = None  # list of (label, crop_bgr)


# ============================================================
# MODEL
# ============================================================
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
    return is_occupied, float(pred), batch


def analyze_image(model, pil_image, rows, cols):
    """Runs grid-based slot detection. Returns annotated image, insights,
    the occupancy grid (rows x cols booleans), and the list of crops."""
    orig_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = orig_cv.shape[:2]
    annotated = orig_cv.copy()

    cell_w = w // cols
    cell_h = h // rows

    total_slots = 0
    occupied_slots = 0
    grid = [[False for _ in range(cols)] for _ in range(rows)]
    crops = []  # (label, crop_bgr) for Grad-CAM selection

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = w if c == cols - 1 else (c + 1) * cell_w
            y2 = h if r == rows - 1 else (r + 1) * cell_h

            crop = orig_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            is_occupied, confidence, _ = classify_slot(model, crop)
            total_slots += 1
            grid[r][c] = bool(is_occupied)
            crops.append((f"Row {r+1}, Col {c+1}", crop))

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
    insights = get_insights(total_slots, occupied_slots)
    return annotated_rgb, insights, grid, crops


def log_to_history(lot_name, insights, rows, cols, grid):
    """Appends this analysis run to a CSV, including the per-slot grid
    (as JSON) so later runs of the same lot layout can be averaged into
    an occupancy heatmap."""
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lot_name": lot_name,
        "rows": rows,
        "cols": cols,
        "total_slots": insights["total"],
        "occupied": insights["occupied"],
        "available": insights["available"],
        "occupancy_pct": insights["occupancy_pct"],
        "congestion": insights["congestion"],
        "grid_json": json.dumps(grid),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORY_CSV):
        df_row.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_CSV, mode="w", header=True, index=False)
    return row


def load_history():
    if os.path.exists(HISTORY_CSV):
        return pd.read_csv(HISTORY_CSV)
    return pd.DataFrame()


# ============================================================
# GRAD-CAM (EXPLAINABLE AI)
# ============================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            continue
    return None


def make_gradcam_heatmap(batch, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(crop_bgr, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (crop_bgr.shape[1], crop_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf_report(lot_name, insights, annotated_pil, timestamp):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ParkVision AI - Parking Analysis Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Lot: {lot_name}", ln=True)
    pdf.cell(0, 8, f"Timestamp: {timestamp}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Total Slots: {insights['total']}", ln=True)
    pdf.cell(0, 8, f"Occupied: {insights['occupied']}", ln=True)
    pdf.cell(0, 8, f"Available: {insights['available']}", ln=True)
    pdf.cell(0, 8, f"Occupancy: {insights['occupancy_pct']}%", ln=True)
    pdf.cell(0, 8, f"Congestion Level: {insights['congestion']}", ln=True)
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Recommendation: {insights['recommendation']}")
    pdf.ln(4)

    img_buffer = io.BytesIO()
    annotated_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    pdf.image(img_buffer, x=10, w=190)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🅿️ ParkVision AI")
    st.caption("Intelligent Urban Parking Analytics & Space Optimisation Platform")
    st.markdown("---")
    st.markdown(
        "**How to use:**\n"
        "1. Name the parking lot you're analyzing\n"
        "2. Upload a parking lot image\n"
        "3. Enter the number of slot rows and columns visible\n"
        "4. Click **Analyze Parking Slots**\n"
        "5. Explore Model Performance, Dataset Info, Heatmap, "
        "History, Video, and Q&A tabs"
    )
    st.markdown("---")
    st.markdown(
        "**Model:** MobileNetV2 (Transfer Learning)\n\n"
        "**Classes:** Occupied / Empty\n\n"
        "**Input size:** 224x224"
    )
    st.markdown("---")
    st.caption("Built for the Machine Learning & Deep Learning course — "
               "Summative Assessment")


# ============================================================
# TABS
# ============================================================
(tab_detect, tab_model, tab_dataset, tab_heatmap,
 tab_history, tab_qa, tab_settings) = st.tabs([
    "🔍 Detect", "📊 Model Performance", "🗂️ Dataset Info",
    "🔥 Heatmap", "🕒 History", "💬 Ask ParkVision", "⚙️ Settings"
])

# ------------------------------------------------------------
# TAB 1: DETECT
# ------------------------------------------------------------
with tab_detect:
    st.markdown("""
    <div class="pv-hero">
        <h1>🅿️ ParkVision AI</h1>
        <p>Upload a parking lot photo — we'll grid it, classify every slot,
        and tell you whether it's worth parking here.</p>
    </div>
    """, unsafe_allow_html=True)

    lot_name = st.text_input("Lot Name", value="Lot A",
                              help="Name this parking lot to track it separately "
                                   "in History, Heatmap, and multi-lot comparison.")

    uploaded_file = st.file_uploader(
        "Upload a parking lot image", type=["jpg", "jpeg", "png"]
    )

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
            with st.spinner("Analyzing parking slots..."):
                model = load_model()
                annotated_rgb, insights, grid, crops = analyze_image(model, pil_image, rows, cols)
                history_row = log_to_history(lot_name, insights, rows, cols, grid)

            st.session_state["latest_insights"] = insights
            st.session_state["latest_lot_name"] = lot_name
            st.session_state["latest_crops"] = crops

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(pil_image, use_container_width=True)
            with col2:
                st.subheader("Annotated Result")
                st.image(annotated_rgb, use_container_width=True)
                st.caption("🟩 Green = Empty   🟥 Red = Occupied")

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Slots", insights["total"])
            m2.metric("Occupied", insights["occupied"])
            m3.metric("Available", insights["available"])
            m4.metric("Occupancy %", f"{insights['occupancy_pct']}%")

            st.subheader(f"Congestion Level: {insights['congestion']}")

            if insights["congestion"] == "High":
                st.error(f"🔴 {insights['recommendation']}")
            elif insights["congestion"] == "Low":
                st.success(f"✅ {insights['recommendation']}")
            else:
                st.warning(f"🟡 {insights['recommendation']}")

            # ---- Grad-CAM explainability ----
            st.divider()
            st.subheader("🔬 Explain a Slot's Prediction (Grad-CAM)")
            slot_labels = [label for label, _ in crops]
            if slot_labels:
                selected_label = st.selectbox("Choose a slot to explain", slot_labels)
                if st.button("Generate Grad-CAM"):
                    selected_crop = dict(crops)[selected_label]
                    try:
                        last_conv = find_last_conv_layer(model)
                        _, _, batch = classify_slot(model, selected_crop)
                        heatmap = make_gradcam_heatmap(batch, model, last_conv)
                        overlay_rgb = overlay_gradcam(selected_crop, heatmap)
                        gc1, gc2 = st.columns(2)
                        with gc1:
                            st.image(cv2.cvtColor(selected_crop, cv2.COLOR_BGR2RGB),
                                     caption=f"Original crop — {selected_label}")
                        with gc2:
                            st.image(overlay_rgb, caption="Grad-CAM heatmap overlay")
                        st.caption("Warmer colors show regions the model focused on "
                                   "most when making its occupied/empty decision.")
                    except Exception as e:
                        st.info(
                            "Grad-CAM could not be generated for this model "
                            f"architecture ({e}). This can happen if the "
                            "convolutional base is nested inside a wrapper layer."
                        )

            # ---- Downloadable reports ----
            st.divider()
            st.subheader("📥 Download Report")

            annotated_pil = Image.fromarray(annotated_rgb)
            img_buffer = io.BytesIO()
            annotated_pil.save(img_buffer, format="PNG")

            pdf_bytes = generate_pdf_report(
                lot_name, insights, annotated_pil, history_row["timestamp"]
            )

            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "Download Annotated Image",
                    data=img_buffer.getvalue(),
                    file_name="parkvision_result.png",
                    mime="image/png",
                )
            with dl2:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="parkvision_report.pdf",
                    mime="application/pdf",
                )
            with dl3:
                report_text = (
                    "ParkVision AI - Analysis Report\n"
                    f"Lot: {lot_name}\n"
                    f"Timestamp: {history_row['timestamp']}\n"
                    f"Total Slots: {insights['total']}\n"
                    f"Occupied: {insights['occupied']}\n"
                    f"Available: {insights['available']}\n"
                    f"Occupancy: {insights['occupancy_pct']}%\n"
                    f"Congestion Level: {insights['congestion']}\n"
                    f"Recommendation: {insights['recommendation']}\n"
                )
                st.download_button(
                    "Download Report (.txt)",
                    data=report_text,
                    file_name="parkvision_report.txt",
                    mime="text/plain",
                )
    else:
        st.info("👆 Upload an image and set rows/columns to get started.")


# ------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# ------------------------------------------------------------
with tab_model:
    render_header("📊 Model Performance",
                   "How the underlying MobileNetV2 classifier performs on unseen data.")

    # --- EDIT THESE VALUES with your actual training results ---
    MODEL_NAME = "MobileNetV2 (Transfer Learning)"
    EPOCHS = 20
    BATCH_SIZE = 16
    TEST_ACCURACY = 0.91
    PRECISION = 0.90
    RECALL = 0.89

    st.markdown(f"""
    | Detail | Value |
    |---|---|
    | Architecture | {MODEL_NAME} |
    | Input size | 224 x 224 |
    | Classes | Occupied, Empty |
    | Epochs | {EPOCHS} |
    | Batch size | {BATCH_SIZE} |
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{TEST_ACCURACY*100:.1f}%")
    c2.metric("Precision", f"{PRECISION*100:.1f}%")
    c3.metric("Recall", f"{RECALL*100:.1f}%")

    st.subheader("Confusion Matrix")
    cm_image_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_image_path):
        st.image(cm_image_path, caption="Confusion Matrix (Test Set)", width=400)
    else:
        st.info(
            "No confusion matrix image found. Generate one in your training "
            f"notebook (e.g. with sklearn.metrics.ConfusionMatrixDisplay) and "
            f"save it as `{cm_image_path}` in your repo to display it here."
        )

    st.subheader("Training Curves")
    curve_image_path = os.path.join(RESULTS_DIR, "training_curves.png")
    if os.path.exists(curve_image_path):
        st.image(curve_image_path, caption="Training / Validation Accuracy & Loss")
    else:
        st.info(
            "No training curve image found. Save your accuracy/loss plot as "
            f"`{curve_image_path}` from your Colab notebook to display it here."
        )


# ------------------------------------------------------------
# TAB 3: DATASET INFO
# ------------------------------------------------------------
with tab_dataset:
    render_header("🗂️ Dataset Information",
                   "Class balance and preprocessing behind the training data.")

    occupied_dir = os.path.join(DATASET_DIR, "occupied")
    empty_dir = os.path.join(DATASET_DIR, "empty")

    if os.path.isdir(occupied_dir) and os.path.isdir(empty_dir):
        n_occupied = len([
            f for f in os.listdir(occupied_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        n_empty = len([
            f for f in os.listdir(empty_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        d1, d2, d3 = st.columns(3)
        d1.metric("Occupied Images", n_occupied)
        d2.metric("Empty Images", n_empty)
        d3.metric("Total Images", n_occupied + n_empty)

        chart_df = pd.DataFrame({
            "Class": ["Occupied", "Empty"],
            "Count": [n_occupied, n_empty],
        }).set_index("Class")
        st.bar_chart(chart_df)
    else:
        st.info(
            "Dataset folders not found in this deployment (this is expected "
            "if you did not upload the full dataset to keep the repo small). "
            "Below is the preprocessing summary used during training:"
        )

    st.markdown("""
    **Preprocessing steps applied:**
    - Source: PKLot dataset (parking lot images under varied weather conditions)
    - Minimum 100 images per class (Occupied / Empty)
    - Resized to 224 x 224 pixels
    - Augmentation: rotation, horizontal flip, brightness adjustment
    - Split: 70% training / 15% validation / 15% testing
    - Organised into `/occupied` and `/empty` class folders
    """)


# ------------------------------------------------------------
# TAB 4: HEATMAP
# ------------------------------------------------------------
with tab_heatmap:
    render_header("🔥 Occupancy Heatmap",
                   "Which slot positions fill up most often, averaged across every logged analysis.")

    history_df = load_history()

    if history_df.empty:
        st.info("No history yet. Run a few analyses on the Detect tab first "
                 "(ideally the same lot, same rows/cols) to build a heatmap.")
    else:
        lot_options = sorted(history_df["lot_name"].dropna().unique().tolist())
        selected_lot = st.selectbox("Select a lot", lot_options)

        lot_df = history_df[history_df["lot_name"] == selected_lot]

        if lot_df.empty:
            st.info("No entries for this lot yet.")
        else:
            # Use the most common grid shape for this lot
            shape_counts = lot_df.groupby(["rows", "cols"]).size().reset_index(name="count")
            best_shape = shape_counts.sort_values("count", ascending=False).iloc[0]
            r_shape, c_shape = int(best_shape["rows"]), int(best_shape["cols"])

            matching_df = lot_df[(lot_df["rows"] == r_shape) & (lot_df["cols"] == c_shape)]

            grids = []
            for grid_json in matching_df["grid_json"]:
                try:
                    grids.append(np.array(json.loads(grid_json), dtype=float))
                except (json.JSONDecodeError, TypeError):
                    continue

            if len(grids) == 0:
                st.info("No valid grid data found for this lot yet.")
            else:
                avg_grid = np.mean(grids, axis=0)  # occupancy rate per cell, 0-1

                st.caption(
                    f"Averaged over {len(grids)} analysis run(s) of '{selected_lot}' "
                    f"with a {r_shape}x{c_shape} slot layout."
                )

                fig, ax = plt.subplots(figsize=(min(2 * c_shape, 10), min(2 * r_shape, 8)))
                im = ax.imshow(avg_grid, cmap="RdYlGn_r", vmin=0, vmax=1)
                ax.set_xticks(range(c_shape))
                ax.set_yticks(range(r_shape))
                ax.set_xticklabels([f"C{c+1}" for c in range(c_shape)])
                ax.set_yticklabels([f"R{r+1}" for r in range(r_shape)])
                for r in range(r_shape):
                    for c in range(c_shape):
                        ax.text(c, r, f"{avg_grid[r][c]*100:.0f}%",
                                ha="center", va="center", fontsize=9,
                                color="black")
                fig.colorbar(im, ax=ax, label="Occupancy rate")
                ax.set_title(f"Occupancy Heatmap — {selected_lot}")
                st.pyplot(fig)


# ------------------------------------------------------------
# TAB 5: HISTORY (incl. multi-lot comparison)
# ------------------------------------------------------------
with tab_history:
    render_header("🕒 Session History & Lot Comparison",
                   "Every analysis you've run, and how your lots stack up against each other.")

    history_df = load_history()

    if history_df.empty:
        st.info("No analyses logged yet. Run a detection in the 'Detect' tab "
                "to start building history.")
    else:
        st.subheader("All Logged Analyses")
        st.dataframe(
            history_df.drop(columns=["grid_json"], errors="ignore"),
            use_container_width=True
        )

        st.subheader("Occupancy % Over Time (per lot)")
        for lot in sorted(history_df["lot_name"].dropna().unique()):
            lot_trend = history_df[history_df["lot_name"] == lot].copy()
            if len(lot_trend) > 1:
                lot_trend = lot_trend.set_index("timestamp")[["occupancy_pct"]]
                st.markdown(f"**{lot}**")
                st.line_chart(lot_trend)

        st.subheader("Compare Lots (Latest Reading)")
        latest_per_lot = (
            history_df.sort_values("timestamp")
            .groupby("lot_name")
            .tail(1)
            .set_index("lot_name")[["occupancy_pct"]]
        )
        st.bar_chart(latest_per_lot)

        st.download_button(
            "Download Full History (.csv)",
            data=history_df.to_csv(index=False),
            file_name="parkvision_history.csv",
            mime="text/csv",
        )


# TAB 7: Q&A ASSISTANT
# ------------------------------------------------------------
with tab_qa:
    render_header("💬 Ask ParkVision",
                   "A smarter assistant that understands availability, trends, comparisons, "
                   "and advice — grounded in your logged history, not just the last result.")

    insights = st.session_state.get("latest_insights")
    lot_name_for_qa = st.session_state.get("latest_lot_name")
    history_df_qa = load_history()

    if insights is None:
        st.info("Run an analysis on the Detect tab first — this assistant "
                 "answers questions about your results.")
    else:
        if "qa_chat" not in st.session_state:
            st.session_state["qa_chat"] = []

        def best_lot_reading(lot_name, history_df):
            """Return the latest row for a lot from history, or None."""
            if history_df.empty or lot_name is None:
                return None
            matches = history_df[history_df["lot_name"] == lot_name]
            if matches.empty:
                return None
            return matches.sort_values("timestamp").iloc[-1]

        def answer_question(q, insights, lot_name, history_df):
            q_lower = q.lower().strip()

            # --- Greeting ---
            if any(w in q_lower for w in ["hi", "hello", "hey"]) and len(q_lower) < 12:
                return (f"Hey! I'm tracking **{lot_name}** right now — "
                        f"{insights['available']} of {insights['total']} slots free. "
                        "Ask me about availability, trends, comparisons, or advice.")

            # --- Availability ---
            if any(w in q_lower for w in ["available", "free", "empty", "open space", "vacant"]):
                if insights["available"] == 0:
                    return (f"No available slots in {lot_name} right now — "
                            "it's completely full. I'd suggest checking another lot.")
                return (f"There are **{insights['available']}** available slot(s) out of "
                        f"{insights['total']} in {lot_name}.")

            # --- Occupancy / fullness ---
            if any(w in q_lower for w in ["occupied", "full", "taken", "how full"]):
                return (f"**{insights['occupied']}** slot(s) are currently occupied "
                        f"in {lot_name} — that's **{insights['occupancy_pct']}%** occupancy.")

            # --- Congestion / busyness ---
            if any(w in q_lower for w in ["congestion", "busy", "crowded", "traffic"]):
                level = insights["congestion"]
                extra = {
                    "Low": "Plenty of room — a good time to head over.",
                    "Moderate": "Filling up, but you should still find a spot.",
                    "High": "Very congested right now — expect a long search for a spot.",
                }.get(level, "")
                return f"Congestion in {lot_name} is currently **{level}**. {extra}"

            # --- Recommendation / advice ---
            if any(w in q_lower for w in ["should i", "park here", "recommend", "advice", "worth it"]):
                return insights["recommendation"]

            # --- Total capacity ---
            if "how many" in q_lower and "total" in q_lower:
                return f"{lot_name} has **{insights['total']}** total slots."

            # --- Trend / change over time ---
            if any(w in q_lower for w in ["trend", "change", "increasing", "decreasing", "over time", "getting worse", "getting better"]):
                if history_df.empty:
                    return "I don't have enough logged history yet to describe a trend — run a few more analyses."
                lot_history = history_df[history_df["lot_name"] == lot_name].sort_values("timestamp")
                if len(lot_history) < 2:
                    return (f"I only have one logged reading for {lot_name} so far "
                            "— analyze it a couple more times to see a trend.")
                first_pct = lot_history.iloc[0]["occupancy_pct"]
                last_pct = lot_history.iloc[-1]["occupancy_pct"]
                diff = last_pct - first_pct
                if diff > 5:
                    direction = f"trending **up** (+{diff:.0f} points) — getting busier"
                elif diff < -5:
                    direction = f"trending **down** ({diff:.0f} points) — getting freer"
                else:
                    direction = "fairly **stable**"
                return (f"Across {len(lot_history)} logged analyses, occupancy at {lot_name} is "
                        f"{direction}, from {first_pct}% to {last_pct}%.")

            # --- Compare two lots ---
            if any(w in q_lower for w in ["compare", "vs", "versus", "better", "which lot"]):
                if history_df.empty or history_df["lot_name"].nunique() < 2:
                    return "I need at least two different lots logged in History to compare them."
                latest_per_lot = (
                    history_df.sort_values("timestamp")
                    .groupby("lot_name")
                    .tail(1)[["lot_name", "occupancy_pct"]]
                    .sort_values("occupancy_pct")
                )
                best = latest_per_lot.iloc[0]
                lines = [f"{r['lot_name']}: {r['occupancy_pct']}% occupied"
                         for _, r in latest_per_lot.iterrows()]
                return ("Here's how your logged lots compare (latest reading):\n\n"
                        + "\n".join(f"- {line}" for line in lines)
                        + f"\n\n**{best['lot_name']}** currently has the most space available.")

            # --- Best/worst time-of-day hint (based on history, if enough data) ---
            if any(w in q_lower for w in ["best time", "when should i", "quietest"]):
                lot_history = history_df[history_df["lot_name"] == lot_name] if not history_df.empty else history_df
                if lot_history is None or lot_history.empty or len(lot_history) < 3:
                    return ("I don't have enough logged readings for this lot yet to spot a "
                            "best time — keep analyzing it periodically and ask again later.")
                quietest = lot_history.sort_values("occupancy_pct").iloc[0]
                return (f"Based on logged history, the quietest recorded reading for {lot_name} "
                        f"was {quietest['occupancy_pct']}% occupancy at {quietest['timestamp']}.")

            # --- Fallback ---
            return (
                "I can help with: **availability**, **occupancy**, **congestion**, "
                "**trends over time**, **comparing lots**, or **whether you should park here**. "
                "Try asking one of those directly."
            )

        # --- Chat-style interaction ---
        question = st.text_input("Ask a question", placeholder="e.g. Is Lot A trending busier today?")

        col_ask, col_clear = st.columns([1, 1])
        with col_ask:
            ask_clicked = st.button("Ask", type="primary")
        with col_clear:
            clear_clicked = st.button("Clear conversation")

        if clear_clicked:
            st.session_state["qa_chat"] = []

        if ask_clicked and question:
            answer = answer_question(question, insights, lot_name_for_qa, history_df_qa)
            st.session_state["qa_chat"].append(("You", question))
            st.session_state["qa_chat"].append(("ParkVision", answer))

        for speaker, msg in st.session_state["qa_chat"][-12:]:
            if speaker == "You":
                st.markdown(f"🙋 **You:** {msg}")
            else:
                st.markdown(f"🅿️ **ParkVision:** {msg}")

        st.divider()
        st.caption(
            f"Currently grounded in the latest analysis of **{lot_name_for_qa}**: "
            f"{insights['occupied']}/{insights['total']} occupied "
            f"({insights['occupancy_pct']}%), congestion: {insights['congestion']}. "
            f"Trend and comparison questions also draw on your full logged History."
        )

# ------------------------------------------------------------
# TAB 8: SETTINGS
# ------------------------------------------------------------
with tab_settings:
    render_header("⚙️ Settings",
                   "Customize how ParkVision AI looks and behaves.")

    st.subheader("🎨 Appearance")
    theme_choice = st.radio(
        "Theme",
        options=["Dark", "Light (Blue)"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1,
        horizontal=True,
        help="Light mode uses a soft blue palette instead of plain white.",
    )

    if theme_choice != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = theme_choice
        st.rerun()

    st.divider()

    st.subheader("🅿️ Default Lot Name")
    st.text_input(
        "Pre-fill this name on the Detect tab",
        value=st.session_state.get("default_lot_name", "Lot A"),
        key="default_lot_name",
        help="Handy if you're usually analyzing the same lot repeatedly.",
    )

    st.divider()

    st.subheader("🗑️ Data Management")
    st.caption("Clear all logged analysis history (cannot be undone).")
    if st.button("Clear All History"):
        if os.path.exists(HISTORY_CSV):
            os.remove(HISTORY_CSV)
            st.success("History cleared. Refresh the History or Heatmap tab to see the change.")
        else:
            st.info("There's no history to clear yet.")

    st.divider()
    st.caption("ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform")
    
