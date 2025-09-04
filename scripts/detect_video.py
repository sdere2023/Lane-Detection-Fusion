#!/usr/bin/env python3
"""
Run YOLO detection on a video and write an annotated copy to outputs/.
Designed to run on CPU/MPS with ONNX weights; .pt also works if available.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Set

import cv2
import torch
from ultralytics import YOLO

# -------- quick knobs --------
ALLOWED: Set[str] = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}

PER_CLASS_CONF: Dict[str, float] = {
    "person": 0.50,        # stricter to avoid ground blobs
    "car": 0.25,
    "truck": 0.30,
    "bus": 0.30,
    "motorcycle": 0.30,
    "bicycle": 0.30,
    "traffic light": 0.30,
    "stop sign": 0.30,
}
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video object detection with basic filtering.")
    p.add_argument("--weights", type=Path, default=Path("models/best.onnx"), help="Path to .pt or .onnx")
    p.add_argument("--source", type=Path, default=Path("inputs/road_demo.mp4"), help="Video file path")
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="Base confidence (fallback)")
    p.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    p.add_argument("--min_area", type=int, default=25 * 25, help="Discard boxes smaller than this area (px^2)")
    p.add_argument("--roi_lower_frac", type=float, default=0.0,
                   help="Keep boxes whose center-Y >= H*frac (e.g., 0.45). 0 disables.")
    p.add_argument("--show", action="store_true", help="Preview while processing")
    p.add_argument("--track", action="store_true",
                   help="Use Ultralytics tracker (annotated video will be saved under runs/track/*)")
    return p.parse_args()


def pick_device() -> str:
    # Ultralytics handles ONNX on CPU transparently; MPS works for .pt on Apple.
    return "mps" if torch.backends.mps.is_available() else "cpu"


def ensure_out_path(inp: Path, outdir: Path, suffix: str = "_det") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{inp.stem}{suffix}{inp.suffix}"


def keep_indices(names: Iterable[str], xyxy, confs, clss, args) -> torch.Tensor:
    """
    Build a boolean mask of boxes to keep according to allowlist, per-class conf,
    min area, and an optional bottom-ROI.
    """
    n = len(confs)
    keep = torch.zeros(n, dtype=torch.bool)
    roi_y = None
    if args.roi_lower_frac > 0.0:
        roi_y = None  # set by caller per-frame to the actual image height if needed
    for i in range(n):
        cls_id = int(clss[i].item())
        name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        if name not in ALLOWED:
            continue
        conf = float(confs[i].item())
        min_conf = PER_CLASS_CONF.get(name, args.conf)
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = map(float, xyxy[i].tolist())
        if (x2 - x1) * (y2 - y1) < args.min_area:
            continue
        if getattr(args, "_roi_y", None) is not None:
            cy = 0.5 * (y1 + y2)
            if cy < args._roi_y:  # type: ignore[attr-defined]
                continue
        keep[i] = True
    return keep


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = pick_device()
    logging.info("Device: %s", device)

    model = YOLO(str(args.weights))
    try:
        model.fuse()  # no-op for ONNX
    except Exception:
        pass

    # Tracking mode delegates writing to Ultralytics
    if args.track:
        logging.info("Track mode active; Ultralytics will manage the output video.")
        res = model.track(
            source=str(args.source),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            tracker="bytetrack.yaml",
            persist=True,
            save=True,
            verbose=False,
        )
        try:
            save_dir = res[-1].save_dir if res else "runs/track"
            logging.info("Done. Check: %s", save_dir)
        except Exception:
            logging.info("Done. Check your latest runs/track/* folder.")
        return

    # Manual loop mode: we control the writer
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = ensure_out_path(args.source, args.outdir, suffix="_det")
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if args.roi_lower_frac > 0.0:
        args._roi_y = int(args.roi_lower_frac * h)  # attach for use in keep_indices
        logging.info("ROI active: keep boxes with center-y >= %d (frac=%.2f)", args._roi_y, args.roi_lower_frac)

    logging.info("Writing to %s", out_path)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes
        if boxes is not None and boxes.shape[0] > 0:
            mask = keep_indices(r.names, boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu().to(torch.int64), args)
            try:
                r.boxes = boxes[mask]
            except Exception:
                r.boxes = boxes if mask.any() else boxes[:0]

        annotated = r.plot()
        writer.write(annotated)

        if args.show:
            cv2.imshow("Preview", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frames += 1
        if frames % 50 == 0:
            logging.info("Processed %d frames...", frames)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logging.info("Saved annotated video to: %s", out_path)


if __name__ == "__main__":
    main()
