# Autonomous Driving Mini (Lane + Object Detection)

A lightweight prototype that overlays **lane detection** (OpenCV, Canny + Hough + smoothing) with **object detection** (YOLOv8 via ONNX).  
Runs on MacBook CPU or Colab GPU. Great as a learning project or research demo.

## 🚀 Features
- Lane detection pipeline with temporal smoothing
- YOLO detections exported to ONNX for CPU inference
- Fusion script overlays both lanes and objects
- Modular structure (`src/mini_drive/`, `scripts/`)
- Works with sample Udacity lane videos (`solidWhiteRight`, `solidYellowLeft`, `challenge`)

## 📦 Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
