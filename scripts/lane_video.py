import argparse, os, cv2
from mini_drive.lanes import process_lane_frame

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="inputs/road_demo.mp4")
    p.add_argument("--out_suffix", type=str, default="_lanes.mp4")
    p.add_argument("--canny1", type=int, default=80)
    p.add_argument("--canny2", type=int, default=160)
    p.add_argument("--hough_thresh", type=int, default=30)
    p.add_argument("--min_line_len", type=int, default=25)
    p.add_argument("--max_line_gap", type=int, default=20)
    p.add_argument("--roi_top_frac", type=float, default=0.60)
    p.add_argument("--roi_bot_frac", type=float, default=0.95)
    p.add_argument("--smooth", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(args.source))
    out_path = os.path.join("outputs", f"{stem}{args.out_suffix}")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    state = {"left": None, "right": None}
    alpha = 1.0 / max(1, args.smooth)
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        vis = process_lane_frame(frame, args.canny1, args.canny2, args.hough_thresh,
                                 args.min_line_len, args.max_line_gap,
                                 args.roi_top_frac, args.roi_bot_frac, alpha, state)
        writer.write(vis)
        frames += 1
        if frames % 50 == 0:
            print(f"Processed {frames} frames...")

    cap.release(); writer.release()
    print(f"Saved lane overlay to: {out_path}")

if __name__ == "__main__":
    main()
