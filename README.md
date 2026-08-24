# ParkVision AI 🚗📷

**Intelligent Urban Parking Analytics & Space Optimisation Platform**

## 📌 About

ParkVision AI is a computer vision-based system that automatically detects and analyzes parking slot occupancy from aerial or CCTV-style parking lot images. Using a MobileNetV2 deep learning model, it divides an uploaded image into a grid of individual parking slots, classifies each one as **Occupied** or **Empty**, and turns that into real-time insights — occupancy percentage, congestion level, and a clear recommendation on whether it's worth parking there.

Beyond single-image detection, the platform tracks multiple parking lots over time, visualizes usage patterns, explains its own predictions, and simulates live video monitoring — making it closer to a real-world smart parking management tool than a one-off classifier demo.

## ✨ Features

### 🔍 Slot Detection
Upload a parking lot image and specify the grid layout (rows x columns). Each slot is classified as occupied or empty with a confidence score, visualized directly on the image — 🟩 green for empty, 🟥 red for occupied.

### 💡 Smart Recommendation Engine
Beyond just showing numbers, the app generates a clear recommendation ("proceed to park" vs "try another location") based on the calculated congestion level (Low / Moderate / High).

### 📊 Model Performance Dashboard
Displays the underlying model's architecture, training parameters, and evaluation metrics — accuracy, precision, recall — along with a confusion matrix and training curves.

### 🗂️ Dataset Transparency
Shows class balance (occupied vs empty image counts) and a summary of the preprocessing pipeline used to prepare the training data, including augmentation and train/validation/test split.

### 🔥 Occupancy Heatmap
Aggregates results across repeated analyses of the same parking lot to reveal which specific slot positions tend to fill up most often.

### 🕒 Multi-Lot History & Comparison
Every analysis is logged with a timestamp and lot name, enabling occupancy trend charts over time per lot, plus a side-by-side comparison across multiple named lots.

### 🎥 Video Analysis (Simulated Live Feed)
Accepts short video clips and samples frames at regular intervals, plotting how occupancy changes over the duration of the clip — approximating a real-time camera feed.

### 🔬 Explainable AI (Grad-CAM)
For any detected slot, generates a Grad-CAM heatmap overlay showing which regions of the image most influenced the model's occupied/empty decision.

### 💬 Ask ParkVision (Q&A Assistant)
A built-in assistant that answers natural-language questions about the most recent analysis — availability, occupancy, congestion level, and parking recommendations.

### 📥 Exportable Reports
One-click export of results as an annotated image, a formatted PDF report, or a plain text summary — ready to share or attach to documentation.

## 🎯 Purpose

Finding an available parking spot in busy lots is often time-consuming, leading to congestion, wasted fuel, and driver frustration. ParkVision AI addresses this using only camera/image input rather than costly per-slot hardware sensors — making occupancy monitoring low-cost, scalable, and deployable with a single overhead camera per lot.

## 🛠️ Tech Stack

| Component            | Technology         |
|-----------------------|---------------------|
| Model / Deep Learning | TensorFlow (MobileNetV2) |
| Frontend / Dashboard  | Streamlit           |
| Image Processing      | OpenCV / NumPy      |
| Data Handling         | Pandas              |
| Visualization         | Matplotlib          |
| Report Export         | fpdf2                |
| Language              | Python              |

## ✅ Conclusion

ParkVision AI demonstrates how computer vision and deep learning can be applied to a practical, everyday problem — parking congestion — using only camera input. By combining slot-level detection with historical tracking, explainability, and multi-lot analytics, the system shows strong potential for real-world deployment in parking lots, malls, and smart city infrastructure.

