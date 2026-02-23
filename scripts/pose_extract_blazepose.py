#!/usr/bin/env python3
import pathlib, sys
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.signal import savgol_filter

# --- Paths ---
CLIPS = pathlib.Path("data/clips")
POSES = pathlib.Path("data/poses"); POSES.mkdir(parents=True, exist_ok=True)
VIS   = pathlib.Path("data/vis");   VIS.mkdir(parents=True, exist_ok=True)

# --- MediaPipe (import lazily to avoid import cost if missing) ---
try:
    import mediapipe as mp
except ImportError:
    sys.exit("ERROR: mediapipe not found. Install with: pip install mediapipe")

mp_drawing = mp.solutions.drawing_utils
mp_pose     = mp.solutions.pose

# --- Config ---
MODEL_COMPLEXITY = 2          # 0/1/2; use 2 for best quality
DETECT_THRESH    = 0.5        # min_detection_confidence
TRACK_THRESH     = 0.5        # min_tracking_confidence
KP_CONF_THRESH   = 0.25       # mask low-confidence kpts for interpolation
SMOOTH_WIN       = 9          # odd integer; tune 7–21
SMOOTH_POLY      = 2

# BlazePose full-body landmarks (33):
BLAZEPOSE_JOINTS = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]
J = len(BLAZEPOSE_JOINTS)  # 33

# Simple set of edges for rendering (subset of MediaPipe’s full topology)
EDGES = [
    (11,13),(13,15),(12,14),(14,16),(11,12),
    (23,25),(25,27),(24,26),(26,28),(23,24),
    (27,29),(28,30),(29,31),(30,32), # lower legs & feet
    (15,17),(17,19),(15,21), (16,18),(18,20),(16,22), # hands (rough)
    (11,23),(12,24), # torso
    (0,11),(0,12),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6), # head/eyes
]

@dataclass
class PosePack:
    kpts: np.ndarray   # [T, 33, 2], pixel coords
    conf: np.ndarray   # [T, 33], landmark visibility
    fps: float
    size: Tuple[int,int]  # (H, W)
    joints: List[str]

def list_clips():
    return sorted(CLIPS.glob("*.mp4"))

def interpolate_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if out.ndim == 2:
        T, J = out.shape
        x = np.arange(T)
        for j in range(J):
            col = out[:, j]
            nans = np.isnan(col)
            if nans.all(): 
                continue
            out[nans, j] = np.interp(x[nans], x[~nans], col[~nans])
    elif out.ndim == 3:
        T, J, C = out.shape
        x = np.arange(T)
        for j in range(J):
            for c in range(C):
                col = out[:, j, c]
                nans = np.isnan(col)
                if nans.all():
                    continue
                out[nans, j, c] = np.interp(x[nans], x[~nans], col[~nans])
    return out

def smooth_series(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] < SMOOTH_WIN:
        return arr
    out = arr.copy()
    if out.ndim == 2:
        for j in range(out.shape[1]):
            out[:, j] = savgol_filter(out[:, j], SMOOTH_WIN, SMOOTH_POLY, axis=0)
    elif out.ndim == 3:
        for j in range(out.shape[1]):
            for c in range(out.shape[2]):
                out[:, j, c] = savgol_filter(out[:, j, c], SMOOTH_WIN, SMOOTH_POLY, axis=0)
    return out

def render_skeleton(frame, kpts_xy, color=(0,255,0), radius=3, thickness=2):
    # draw points
    for (x,y) in kpts_xy:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x),int(y)), radius, color, -1)
    # draw edges
    for a,b in EDGES:
        xa,ya = kpts_xy[a]; xb,yb = kpts_xy[b]
        if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
            cv2.line(frame, (int(xa),int(ya)), (int(xb),int(yb)), color, thickness)
    return frame

def process_video(clip_path: pathlib.Path, pose_estimator: mp_pose.Pose):
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    k_all, c_all = [], []

    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose_estimator.process(rgb)

        if not res.pose_landmarks:
            k_all.append(np.full((J,2), np.nan, dtype=np.float32))
            c_all.append(np.full((J,), np.nan, dtype=np.float32))
            continue

        lm = res.pose_landmarks.landmark  # 33 landmarks
        k_xy = np.zeros((J,2), dtype=np.float32)
        conf = np.zeros((J,), dtype=np.float32)

        for i, l in enumerate(lm):
            # mp gives normalized coordinates; convert to px
            x = l.x * W
            y = l.y * H
            v = float(l.visibility)  # 0..1 approx. visibility/confidence
            if v < KP_CONF_THRESH or x < 0 or y < 0 or x >= W or y >= H:
                k_xy[i] = np.nan
            else:
                k_xy[i] = (x, y)
            conf[i] = v

        k_all.append(k_xy)
        c_all.append(conf)

    cap.release()
    kpts = np.stack(k_all, axis=0)     # [T,33,2]
    conf = np.stack(c_all, axis=0)     # [T,33]

    # Fill gaps then smooth
    kpts = interpolate_nan(kpts)
    kpts = smooth_series(kpts)

    pack = PosePack(kpts=kpts, conf=conf, fps=fps, size=(H,W), joints=BLAZEPOSE_JOINTS)

    # Save NPZ
    vid = clip_path.stem
    np.savez_compressed(
        POSES / f"{vid}.npz",
        kpts=pack.kpts, conf=pack.conf, fps=pack.fps,
        H=H, W=W, joints=np.array(pack.joints, dtype=object)
    )

    # Optional CSV
    csv_path = POSES / f"{vid}.csv"
    with open(csv_path, "w") as f:
        header = []
        for jn in BLAZEPOSE_JOINTS: header += [f"{jn}_x", f"{jn}_y"]
        header += [f"{jn}_conf" for jn in BLAZEPOSE_JOINTS]
        f.write(",".join(header) + "\n")
        T, JJ, _ = kpts.shape
        for t in range(T):
            row = []
            for j in range(JJ):
                x, y = kpts[t,j]
                row.extend([f"{x:.3f}", f"{y:.3f}"])
            for j in range(JJ):
                row.append(f"{conf[t,j]:.3f}")
            f.write(",".join(row) + "\n")

    # Render overlay for QA
    cap = cv2.VideoCapture(str(clip_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIS / f"{vid}_pose.mp4"), fourcc, pack.fps, (W,H))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = render_skeleton(frame, pack.kpts[t])
        out.write(frame); t += 1
    cap.release(); out.release()

    print(f"✓ Pose saved: {POSES/(vid+'.npz')} | overlay: {VIS/(vid+'_pose.mp4')}")

def main():
    clips = list_clips()
    if not clips:
        print("No clips found in data/clips/*.mp4"); return

    # Create estimator once (single-person tracking)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=DETECT_THRESH,
        min_tracking_confidence=TRACK_THRESH,
        smooth_landmarks=True
    ) as pose_estimator:
        for p in clips:
            print(f"\n=== {p.name} ===")
            process_video(p, pose_estimator)

if __name__ == "__main__":
    main()