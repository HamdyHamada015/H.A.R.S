# H.A.R.S

# 🚗 HARS — Hazard Awareness & Risk System

### Intelligent Road Safety & Collision Risk Detection

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222?style=for-the-badge&logo=github)](YOUR_GITHUB_PAGES_URL)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-111?style=for-the-badge)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

> **HARS** is an AI-powered Computer Vision project designed to detect surrounding objects, estimate their distance, evaluate collision risk, and provide real-time safety warnings.

---

## 🌐 Project Website

### 🚀 Live Website

**[Visit HARS Website →](YOUR_GITHUB_PAGES_URL)**

The GitHub Pages website provides an interactive overview of the project, including:

- Project vision
- System architecture
- Computer Vision pipeline
- Distance estimation
- Risk assessment
- Project screenshots
- Demo results
- Technologies
- Team members
- Future improvements

---

# 🎯 About HARS

HARS stands for:

> **Hazard Awareness & Risk System**

The project focuses on improving road-awareness systems using Artificial Intelligence and Computer Vision.

Instead of simply detecting objects, HARS attempts to answer a more important question:

> **"How dangerous is the detected object to the vehicle?"**

The system combines object detection, distance estimation, object tracking, and risk analysis to generate an understandable safety status.

---

# 🧠 How It Works

```text
             CAMERA / VIDEO
                    │
                    ▼
           ┌─────────────────┐
           │ Frame Processing│
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  YOLO Detection │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Object Tracking │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │    Distance     │
           │    Estimation   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Risk Engine   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Safety Warning  │
           └─────────────────┘

