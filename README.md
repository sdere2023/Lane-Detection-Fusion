# Autonomous Driving Mini (Lane + Object Detection)

A lightweight prototype that overlays **lane detection** (OpenCV, Canny + Hough + smoothing) with **object detection** (YOLOv8 via ONNX).  
Runs on MacBook CPU or Colab GPU. Great as a learning project or research demo.

## 🚀 Features
- Lane detection pipeline with temporal smoothing
- YOLO detections exported to ONNX for CPU inference
- Fusion script overlays both lanes and objects
- Modular structure (`src/mini_drive/`, `scripts/`)
- Works with sample Udacity lane videos (`solidWhiteRight`, `solidYellowLeft`, `challenge`)

## 🛠️ Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🎥 Sample Inputs & Outputs

### 📥 Input Videos
| Video               | Download Link |
|---------------------|---------------|
| solidWhiteRight.mp4  | [Download](https://drive.google.com/uc?export=download&id=1wKz1jC-TK66DS5T1T0m-Jsy4jZE4zxRe) |
| solidYellowLeft.mp4  | [Download](https://drive.google.com/uc?export=download&id=1PIWw4B_CViOFSsXVnnEvvYq-J1x-83k3) |
| road_demo.mp4        | [Download](https://drive.google.com/uc?export=download&id=1ERTPfN41CXrtZ4JghRt2eK8HWgaDl4Wq) |
| challenge.mp4        | [Download](https://drive.google.com/uc?export=download&id=1cAxvty2XeKWVY7Lx9NCgMT7eJfNEI6fd) |

---

### 📤 Example Outputs
| Result Video             | Preview / Download |
|---------------------------|--------------------|
| solidWhiteRight_fusion.mp4   | [Download](https://drive.google.com/uc?export=download&id=1EYM_GLkJBFP9JzpXwk77z8hoeduNXxdv) |
| solidYellowLeft_fusion.mp4   | [Download](https://drive.google.com/uc?export=download&id=1D74gs1InLAVbB0nfknepgwuvNLdgI0k_) |
| road_demo_annotated.mp4      | [Download](https://drive.google.com/uc?export=download&id=1OIVmjj0ZStSj6LjEgiwxDFISMouYtIV_) |
| challenge_fusion.mp4         | [Download](https://drive.google.com/uc?export=download&id=19OWn9chuFpRJyGsXBqC9kG5B62VeyukQ) |
