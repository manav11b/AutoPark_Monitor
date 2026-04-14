# 🚗 AutoPark Monitor

**AutoPark Monitor** is a lightweight computer-vision–based parking slot occupancy detection system designed for **aerial and top-down CCTV footage**. Unlike traditional object-detection approaches that rely on deep learning models like YOLO, this system uses a fast **pixel-based multi-signal detection pipeline** that runs efficiently on CPU-only environments.

It is ideal for smart campus parking, surveillance-based parking analytics, research experiments, and edge-device deployments.

---

## ✨ Features

* 📡 Designed for aerial / top-down parking footage
* ⚡ Runs fully on CPU (no GPU required)
* 🚫 No YOLO or heavy deep learning dependencies
* 🧠 Multi-signal detection pipeline
* 📍 Polygon-based parking slot annotation tool included
* 🔄 Adaptive background modeling per slot
* 🎯 Temporal smoothing to reduce flicker from shadows
* 📊 Real-time occupancy visualization with live statistics overlay

---

## 🏗️ Detection Pipeline

AutoPark Monitor uses a two-stage hybrid detection architecture:

**Stage 1 – Pixel Pre-filter**

* Detects grayscale variation
* Measures edge density
* Quickly skips clearly empty slots

**Stage 2 – Aerial Slot Detector**
Combines four visual signals:

* Foreground deviation from background
* Texture variation (standard deviation)
* HSV saturation information
* Edge density

Each slot maintains its own adaptive background model for improved robustness under lighting changes.

---

## 📂 Project Structure

```
AutoPark_Monitor/
│
├── main.py
├── src/
│   ├── aerial_detector.py
│   ├── pixel_detector.py
│   ├── slot_manager.py
│   ├── polygon_utils.py
│   ├── video_loader.py
│   └── visualization.py
│
├── tools/
│   └── slot_annotation_tool.py
│
├── data/
│   └── slots.json
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```
git clone https://github.com/manav11b/AutoPark_Monitor
cd autopark_monitor
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the system:

```
python main.py --video data/parking_video.mp4 --slots data/slots.json
```

### Controls

| Key   | Action                       |
| ----- | ---------------------------- |
| SPACE | Pause / Resume               |
| + / - | Adjust detection sensitivity |
| R     | Reset background model       |
| D     | Toggle score debug overlay   |
| S     | Save snapshot                |
| Q     | Quit                         |

---

## 🧰 Tech Stack

* Python
* OpenCV
* NumPy
* Shapely

---

## 📌 Applications

* Smart parking monitoring
* Campus parking analytics
* Drone-based parking observation
* Edge-device deployment
* Computer vision research projects

---

## 🔮 Future Improvements

* Automatic parking slot detection
* Multi-camera support
* Web dashboard integration
* Cloud deployment support

---

## 👨‍💻 Author

Developed by **Manav Borkar**
