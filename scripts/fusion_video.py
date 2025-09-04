import argparse, os, time, cv2
from ultralytics import YOLO
from mini_drive.lanes import process_lane_frame

ALLOWED = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}
PER_CLASS_CONF = {"person":0.50,"car":0.25,"truck":0.30,"bus":0.30,"motorcycle":0.30,"bicycle":0.30,"traffic light":0.30,"stop sign":0.30}
MIN_AREA = 25*25

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="models/best.onnx")
    p.add_argument("--source",  type=str, default="inputs/solidWhiteRight.mp4")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.60)
    p.add_argument("--canny1", type=int, default=80)
    p.add_argument("--canny2", type=int, default=160)
    p.add_argument("--hough_thresh", type=int, default=30)
    p.add_argument("--min_line_len", type=int, default=25)
    p.add_argument("--max_line_gap", type=int, default=20)
    p.add_argument("--roi_top_frac", type=float, default=0.60)
    p.add_argument("--roi_bot_frac", type=float, default=0.95)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument("--track", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(args.source))
    out_path = os.path.join("outputs", f"{stem}_fusion{ext}")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    state = {"left": None, "right": None}
    alpha = 1.0 / max(1, args.smooth)

    t0 = time.time(); frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        fused = process_lane_frame(frame, args.canny1, args.canny2, args.hough_thresh,
                                   args.min_line_len, args.max_line_gap,
                                   args.roi_top_frac, args.roi_bot_frac, alpha, state)

        if args.track:
            results = model.track(fused, conf=args.conf, iou=args.iou, imgsz=args.imgsz, persist=True, verbose=False)
        else:
            results = model.predict(fused, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)

        res = results[0]
        boxes = res.boxes
        if boxes is not None and boxes.data.shape[0] > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            keep = []
            for i, (x1,y1,x2,y2) in enumerate(xyxy):
                w = max(0, x2-x1); h = max(0, y2-y1)
                if w*h < MIN_AREA: 
                    continue
                name = res.names[int(cls[i])] if isinstance(res.names, dict) else str(cls[i])
                if name not in ALLOWED: 
                    continue
                if conf[i] >= PER_CLASS_CONF.get(name, args.conf):
                    keep.append(i)

            for i in keep:
                x1,y1,x2,y2 = xyxy[i]
                name = res.names[int(cls[i])] if isinstance(res.names, dict) else str(cls[i])
                label = f"{name} {conf[i]:.2f}"
                cv2.rectangle(fused, (x1,y1), (x2,y2), (0,200,255), 2)
                cv2.putText(fused, label, (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2, cv2.LINE_AA)

        frames += 1
        if frames % 10 == 0:
            fps_live = frames / max(time.time() - t0, 1e-3)
            cv2.putText(fused, f"FPS: {fps_live:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        writer.write(fused)
        if frames % 100 == 0:
            print(f"Processed {frames} frames...")

    cap.release(); writer.release()
    print(f"Saved fusion video -> {out_path}")

if __name__ == "__main__":
    main()
