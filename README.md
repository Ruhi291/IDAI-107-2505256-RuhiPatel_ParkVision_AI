# ParkVision AI 🚗📷

**AI-powered parking occupancy detection using computer vision and deep learning, visualized through a real-time Streamlit dashboard.**

---

## 📌 Overview

ParkVision AI is a computer vision-based system that automatically detects and monitors the occupancy status of parking slots from aerial or CCTV-style parking lot images. Instead of manually checking which spots are free, the system analyzes an image of the lot, divides it into a grid of individual parking slots, and classifies each slot as **Occupied** or **Empty** using a trained deep learning model — displaying the results instantly on an interactive dashboard.

## 🎯 Purpose

Finding an available parking spot in busy lots — malls, offices, campuses, stadiums — is often time-consuming and inefficient, leading to traffic congestion, wasted fuel, and driver frustration. Most existing parking systems rely on manual monitoring or expensive sensor-based hardware installed under every single spot.

ParkVision AI addresses this by using only camera/image input and computer vision, making occupancy detection:
- **Low-cost** — no per-slot hardware sensors required
- **Scalable** — works with a single overhead camera per lot
- **Real-time** — instant occupancy status and congestion insights

## ⚙️ How It Works

1. **Input**: An aerial/CCTV image of the parking lot is provided to the app.
2. **Slot Detection**: The lot is divided into a grid of individual parking slots.
3. **Classification**: A TensorFlow-based deep learning model analyzes each slot and predicts whether it is occupied or empty, along with a confidence score.
4. **Visualization**: Slots are color-coded on the image —
   - 🟩 **Green** = Empty
   - 🟥 **Red** = Occupied
5. **Dashboard Metrics**: The app calculates and displays:
   - Total Slots
   - Occupied Slots
   - Available Slots
   - Occupancy Percentage
   - Congestion Level (e.g. Low / Moderate / High)
6. **Alerts**: A message is shown when the lot is filling up quickly, helping users or lot managers plan ahead.

## 🛠️ Tech Stack

| Component            | Technology         |
|-----------------------|---------------------|
| Model / Deep Learning | TensorFlow          |
| Frontend / Dashboard  | Streamlit           |
| Image Processing      | OpenCV / NumPy      |
| Language              | Python              |

## 📂 Project Structure

```
ParkVision_AI/
├── app.py               # Main Streamlit application
├── src/                 # Source code (preprocessing, model utilities, etc.)
├── models/               # Trained model files
├── results/              # Output images / evaluation results
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x installed
- pip package manager

### Installation
```bash
git clone https://github.com/<your-username>/ParkVision_AI.git
cd ParkVision_AI
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser.

## 📊 Results

The model successfully classifies parking slots as occupied or empty with high confidence in most cases, achieving [add your accuracy/metric here, e.g. "92% classification accuracy on the test set"]. The dashboard provides a clear, real-time snapshot of lot occupancy that could support smart parking management systems.

## ✅ Conclusion

ParkVision AI demonstrates how computer vision and deep learning can be applied to solve a practical, everyday problem — parking congestion — using only camera input rather than costly hardware sensors. By automatically detecting occupancy at the slot level and presenting it through an intuitive dashboard, the system shows strong potential for real-world deployment in parking lots, malls, and smart city infrastructure, helping reduce search time, congestion, and fuel wastage.




This project is for educational/academic purposes. [Add license here if applicable, e.g. MIT License.]
