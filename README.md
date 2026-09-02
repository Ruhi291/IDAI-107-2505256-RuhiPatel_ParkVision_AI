# ParkVision AI 
**Intelligent Urban Parking Analytics & Space Optimisation Platform**

**Live App:** [PASTE YOUR STREAMLIT CLOUD LINK HERE]
**GitHub Repository:** [PASTE YOUR REPO LINK HERE]

---

##  What This App Is

ParkVision AI is a computer vision-based system that automatically detects and analyzes parking slot occupancy from parking lot images. Using a MobileNetV2 deep learning model, it divides an uploaded image into a grid of individual parking slots, classifies each one as **Occupied** or **Empty**, and turns that into real-time insights — occupancy percentage, congestion level, and a clear recommendation on whether it's worth parking there.

Beyond single-image detection, the platform tracks multiple parking lots over time, visualizes usage patterns, explains its own predictions, and lets users ask questions about parking status in plain language — making it closer to a real-world smart parking management tool than a one-off classifier demo.

---

##  Features Explained

###  Slot Detection
The core of the app. Users upload a parking lot image and specify how many rows and columns of slots are visible. The app slices the image into that grid, runs each slot through the trained model, and overlays the result directly on the image — 🟩 green boxes for empty slots, 🟥 red boxes for occupied ones, each with a confidence percentage. This gives an instant, visual answer to "where can I park?" without the user having to inspect the photo themselves.

###  Smart Recommendation Engine
Numbers alone aren't very useful to a driver in a hurry, so the app converts occupancy percentage into a congestion level (Low / Moderate / High) and a plain-language recommendation — either "proceed to park" or "try another location." This turns a technical prediction into an actual decision-making aid.

###  Model Performance Dashboard
Shows the model's architecture, training parameters, and evaluation metrics (accuracy, precision, recall), along with a confusion matrix and training curves. This exists so the AI isn't a "black box" — anyone reviewing the app can see exactly how well the underlying model performs and on what basis its predictions can be trusted.

###  Dataset Transparency
Displays class balance (how many occupied vs. empty training images were used) and a summary of the preprocessing pipeline — resizing, augmentation, and the train/validation/test split. This makes the project's data foundation visible rather than hidden behind the model.

### Occupancy Heatmap
Aggregates results across repeated analyses of the same parking lot to reveal which specific slot positions tend to fill up most often. Over time, this could help a lot manager understand usage patterns — for example, spotting that slots near an entrance are almost always full while further ones are consistently empty.

###  Multi-Lot History & Comparison
Every analysis is logged with a timestamp and lot name, enabling occupancy trend charts over time per lot, plus a side-by-side comparison across multiple named lots. This is what turns the app from a single-photo tool into something that can monitor several locations over time, the way a real city-wide system would need to.

###  Explainable AI (Grad-CAM)
For any detected slot, the app can generate a Grad-CAM heatmap showing which parts of the image most influenced the model's decision. This matters because a model that just says "occupied" or "empty" gives no way to verify *why* — Grad-CAM makes the reasoning visible and helps catch cases where the model might be focusing on the wrong part of the image.

### Ask ParkVision (Q&A Assistant)
A built-in conversational assistant that answers natural-language questions about parking status — availability, occupancy, congestion, trends over time, and comparisons across logged lots. Instead of requiring users to interpret charts themselves, they can simply ask "is Lot A getting busier?" and get a direct answer.

###  Settings & Theming
A dedicated settings panel lets users switch between a dark theme and a light blue theme, set a default lot name for convenience, and clear logged history. This is a usability layer — letting the app adapt to different preferences rather than forcing one fixed look.

###  Exportable Reports
One-click export of results as an annotated image, a formatted PDF report, or a plain text summary — ready to share, print, or attach to documentation without needing to take a screenshot manually.

---

##  Why This App Is Important and Useful

Finding an available parking spot in busy areas is a small but widespread daily frustration — drivers can spend several minutes circling a lot, which adds to traffic congestion, wastes fuel, and increases stress for no productive reason. Traditional solutions to this problem rely on physical sensors installed under every single parking spot, which is expensive to install and maintain at scale.

ParkVision AI solves the same problem using only a camera and computer vision — no hardware sensors required. This makes it:

- **Low-cost** — a single overhead camera can monitor an entire lot, instead of per-slot hardware
- **Scalable** — the same approach can be deployed across many lots in a city without significant added infrastructure
- **Actionable, not just informative** — it doesn't just detect occupancy, it tells the user what to do about it
- **Transparent** — through the Model Performance and Grad-CAM features, the system's reliability and reasoning are visible rather than hidden
- **Usable by non-technical people** — the recommendation engine and Q&A assistant mean a driver doesn't need to understand AI to benefit from it

In the context of smart city infrastructure, this kind of system directly supports reduced congestion, lower emissions from unnecessary circling, and a better overall urban mobility experience.

---

##  Conclusion

ParkVision AI demonstrates how computer vision and deep learning can be applied to a practical, everyday problem — parking congestion — using only camera input rather than costly per-slot hardware sensors. By combining slot-level detection with historical tracking, explainability, multi-lot analytics, and a conversational interface, the system moves beyond a simple classification demo into something that resembles a real, usable smart parking tool. It shows strong potential for real-world deployment in parking lots, malls, and smart city infrastructure, while remaining transparent about its own performance and limitations.

---



