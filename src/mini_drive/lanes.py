# src/mini_drive/lanes.py
import cv2, numpy as np

def region_of_interest_mask(shape, top_frac=0.6, bot_frac=0.95):
    h, w = shape[:2]
    top_y = int(top_frac * h); bot_y = int(bot_frac * h)
    pts = np.array([[
        (int(0.12*w), bot_y), (int(0.45*w), top_y),
        (int(0.55*w), top_y), (int(0.88*w), bot_y)
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask, pts[0]

def average_lane_from_segments(segments, img_shape, slope_sign):
    if not segments: return None
    h, _ = img_shape[:2]
    xs, ys = [], []
    for (x1,y1,x2,y2) in segments:
        if x2 == x1: 
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope_sign == 'neg' and slope >= -0.2: continue
        if slope_sign == 'pos' and slope <=  0.2: continue
        xs += [x1,x2]; ys += [y1,y2]
    if len(xs) < 2: return None
    m, b = np.polyfit(xs, ys, 1)
    if abs(m) < 1e-3: return None
    y1 = int(h*0.95); y2 = int(h*0.60)
    x1 = int((y1 - b) / m); x2 = int((y2 - b) / m)
    return (x1,y1,x2,y2)

def ema_update(prev, new, alpha=0.25):
    if new is None: return prev
    if prev is None: return new
    return tuple(int(p*(1-alpha) + n*alpha) for p, n in zip(prev, new))

def draw_lane_overlay(base_bgr, left, right, roi_poly=None):
    overlay = base_bgr.copy()
    if roi_poly is not None:
        cv2.polylines(overlay, [roi_poly], isClosed=True, color=(255,0,0), thickness=2)
    if left:  cv2.line(overlay, (left[0], left[1]), (left[2], left[3]), (0,255,0), 8)
    if right: cv2.line(overlay, (right[0], right[1]), (right[2], right[3]), (0,255,0), 8)
    if left and right:
        poly = np.array([[left[0],left[1]],[left[2],left[3]],[right[2],right[3]],[right[0],right[1]]], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (0,255,0))
    return cv2.addWeighted(overlay, 0.35, base_bgr, 0.65, 0)

def process_lane_frame(frame, canny1, canny2, hough_thresh, min_line_len, max_line_gap, roi_top_frac, roi_bot_frac, alpha, state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)

    mask, roi_poly = region_of_interest_mask(frame.shape, roi_top_frac, roi_bot_frac)
    masked = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    left_segs, right_segs = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if x2 == x1: 
                continue
            slope = (y2 - y1) / (x2 - x1)
            (left_segs if slope < 0 else right_segs).append((int(x1),int(y1),int(x2),int(y2)))

    left  = average_lane_from_segments(left_segs,  frame.shape, 'neg')
    right = average_lane_from_segments(right_segs, frame.shape, 'pos')

    state["left"]  = ema_update(state.get("left"),  left,  alpha) if left  else state.get("left")
    state["right"] = ema_update(state.get("right"), right, alpha) if right else state.get("right")

    fused = draw_lane_overlay(frame, state.get("left"), state.get("right"), roi_poly)
    return fused
