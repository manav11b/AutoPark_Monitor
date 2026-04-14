<div align="center">

# 🚗 AutoPark Monitor

**Smart parking-slot occupancy detection for aerial & top-down CCTV footage**

*Optimized for CPU-only environments — no GPU, no deep-learning overhead*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![CPU Friendly](https://img.shields.io/badge/CPU-Friendly-22c55e?style=flat-square)](#)
[![No GPU Required](https://img.shields.io/badge/GPU-Not%20Required-f97316?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**AutoPark Monitor** is a lightweight computer-vision system for detecting parking-slot occupancy in aerial and overhead parking camera footage.

Instead of relying on heavy deep-learning detectors like YOLO — which perform poorly on top-down views — AutoPark Monitor uses a **multi-signal pixel-analysis pipeline** that runs efficiently on standard CPU machines.

> Perfect for smart campus parking, drone-based monitoring, edge-device deployments, and CV research.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🎯 **Aerial-first design** | Purpose-built for top-down parking footage |
| ⚡ **CPU-only runtime** | No GPU or CUDA dependencies required |
| 🧠 **No model downloads** | Zero deep-learning model files needed |
| 🔄 **Adaptive background modeling** | Per-slot background model for lighting stability |
| 📡 **Multi-signal pipeline** | Foreground, texture, HSV saturation, and edge signals |
| 🕐 **Temporal smoothing** | Shadow-resistant stability across frames |
| 🖊️ **Interactive annotation tool** | Draw and save slot polygons manually |
| 📊 **Real-time visualization** | Live stats overlay on video feed |

---

## 🧠 Detection Architecture

AutoPark Monitor uses a **two-stage hybrid detection pipeline**:

```
Frame Input
    │
    ▼
┌─────────────────────┐
│   Pixel Pre-Filter  │  ← grayscale variation + edge density
│  (fast early-exit)  │     eliminates empty slots quickly
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────┐
│  Aerial Slot Detector    │  ← 4-signal per-slot analysis
│                          │
│  • Foreground deviation  │  detects presence vs. asphalt
│  • Texture variation     │  captures vehicle surface contrast
│  • HSV saturation        │  highlights colored vehicles
│  • Edge density          │  captures vehicle boundaries
└──────────┬───────────────┘
           │
           ▼
┌─────────────────────┐
│  Temporal Smoothing │  ← rolling window vote for stability
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│  Visualization       │  ← live overlay with occupancy stats
└──────────────────────┘
```

> Each slot maintains its own **adaptive background model**, making the system robust to gradual lighting changes, shadow movement, and environmental shifts.

---

## 📂 Project Structure

```
AutoPark_Monitor/
│
├── main.py                        # Entry point
├── src/
│   ├── aerial_detector.py         # Core 4-signal occupancy detector
│   ├── pixel_detector.py          # Stage 1 fast pixel pre-filter
│   ├── slot_manager.py            # Slot state + background model management
│   ├── polygon_utils.py           # Polygon masking & geometry utilities
│   ├── video_loader.py            # Video capture abstraction
│   └── visualization.py          # Real-time overlay & stats rendering
│
├── tools/
│   └── slot_annotation_tool.py    # Interactive slot polygon annotation
│
├── data/
│   └── slots.json                 # Slot definitions (output of annotation tool)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/manav11b/AutoPark_Monitor
cd AutoPark_Monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Annotate parking slots *(first-time setup)*

```bash
python tools/slot_annotation_tool.py --video data/parking_video.mp4
```

Draw polygons over each parking slot and save them to `data/slots.json`.

### 4. Run the detector

```bash
python main.py --video data/parking_video.mp4 --slots data/slots.json
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `+` / `-` | Adjust detection sensitivity |
| `R` | Reset background model |
| `D` | Toggle debug overlay |
| `S` | Save snapshot |
| `Q` | Quit |

---

## ⚠️ Engineering Decisions & Challenges

### 🚫 Why Not YOLO?

YOLO models are trained primarily on **street-level vehicle perspectives**. In aerial parking footage:

- Vehicles appear as flat rectangles with no perspective cues
- Detection confidence drops dramatically
- Bounding box regression breaks for top-down orientations

AutoPark Monitor uses a **custom aerial pixel-analysis detector** instead — making it lighter, faster, and significantly more reliable for overhead footage.

### 📉 Limited CCTV Dataset Availability

High-quality parking CCTV datasets are mostly paid or restricted. Freely available datasets often suffer from unstable camera angles, low resolution, compression artifacts, and inconsistent lighting. Development relied on available aerial parking recordings to work around these constraints.

### 🌤️ Lighting & Shadow Robustness

Outdoor parking detection is heavily affected by shadows, reflections, and brightness variation throughout the day. AutoPark Monitor addresses this through:

- **Adaptive per-slot background modeling** — each slot learns its own baseline independently
- **Temporal smoothing** — votes across a rolling time window to suppress transient noise
- **Multi-signal fusion** — no single signal dominates; all four must agree

---

## 📊 Applications

- 🎓 Smart campus parking management
- 🛰️ Drone-based aerial parking analytics
- ⚡ Edge-device occupancy detection
- 🔬 Computer-vision research with aerial datasets
- 📹 Surveillance-based parking monitoring

---

## 🔮 Roadmap

- [ ] Automatic slot boundary detection
- [ ] Multi-camera support
- [ ] Web dashboard interface
- [ ] Live CCTV stream integration
- [ ] Cloud deployment pipeline
- [ ] Hybrid deep-learning variant for ground-level cameras

---

## 👨‍💻 Author

**Manav Borkar**
B.Tech Artificial Intelligence · G.H. Raisoni College of Engineering & Management, Nagpur

[![GitHub](https://img.shields.io/badge/GitHub-manav11b-181717?style=flat-square&logo=github)](https://github.com/manav11b)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/manav-borkar)

---

<div align="center">
  <sub>Built with ❤️ for overhead parking detection · No GPU harmed in the making of this project</sub>
</div>
