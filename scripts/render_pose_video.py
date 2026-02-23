# scripts/render_pose_video.py
# Render 2D poses to video with robust scaling, optional beat blink,
# motion magnification around a reference, and hip-centering.
# Skeleton edges are derived via named joints (meta.json).

import argparse, json
from pathlib import Path
import cv2
import numpy as np

# ========================== Schemas (by NAME) ==========================

# MediaPipe BlazePose 33 joint names (canonical order)
BLAZEPOSE_33_NAMES = [
    "nose",
    "left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear",
    "mouth_left","mouth_right",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_pinky","right_pinky",
    "left_index","right_index",
    "left_thumb","right_thumb",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

# Minimal, visually stable edges (by NAME)
BLAZEPOSE_33_EDGES_BY_NAME = [
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_shoulder","right_shoulder"),
    ("left_hip","right_hip"),
    ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ("left_wrist","left_index"), ("left_index","left_pinky"),
    ("right_wrist","right_index"), ("right_index","right_pinky"),
    ("left_ankle","left_heel"), ("left_heel","left_foot_index"),
    ("right_ankle","right_heel"), ("right_heel","right_foot_index"),
]

SCHEMAS = {
    "blazepose": {
        "names": BLAZEPOSE_33_NAMES,
        "edges_by_name": BLAZEPOSE_33_EDGES_BY_NAME,
    }
}

ALIASES = {
    # common aliasing: map dataset names to our canonical ones if needed
    "l_shoulder": "left_shoulder", "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow", "r_elbow": "right_elbow",
    "l_wrist": "left_wrist", "r_wrist": "right_wrist",
    "l_hip": "left_hip", "r_hip": "right_hip",
    "l_knee": "left_knee", "r_knee": "right_knee",
    "l_ankle": "left_ankle", "r_ankle": "right_ankle",
    "l_heel": "left_heel", "r_heel": "right_heel",
    "l_foot_index": "left_foot_index", "r_foot_index": "right_foot_index",
    "l_index": "left_index", "r_index": "right_index",
    "l_pinky": "left_pinky", "r_pinky": "right_pinky",
}

# ============================== IO ==============================

def _to_vec(x) -> np.ndarray:
    """Convert nested JSON objects into a flat 1D float vector."""
    if isinstance(x, dict):
        for k in ("pose","P","data","keypoints","kpts","joints","coords"):
            if k in x:
                x = x[k]
                break
        else:
            x = [v for _, v in sorted(x.items())]
    a = np.array(x, dtype=np.float32)
    return a.reshape(-1) if a.ndim > 1 else a

def load_pose_any(path: Path) -> np.ndarray:
    """Load poses from .jsonl/.json/.npy/.npz → float32 [T, D]."""
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                s = line.strip()
                if not s: continue
                obj = json.loads(s)
                if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[1], (list, tuple, dict)):
                    vec = _to_vec(obj[1])
                else:
                    vec = _to_vec(obj)
                rows.append(vec)
        if not rows: return np.zeros((0, 0), dtype=np.float32)
        D = max(r.shape[0] for r in rows)
        out = np.zeros((len(rows), D), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
        return out

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for k in ("poses","P","data","keypoints","kpts","joints","coords"):
                if k in data:
                    arr = np.asarray(data[k], dtype=np.float32); break
            else:
                arr = np.asarray(list(data.values()), dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)
        return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    if path.suffix == ".npz":
        z = np.load(path)
        key = next((k for k in ("P","poses","pose","arr_0") if k in z.files), z.files[0])
        arr = z[key].astype(np.float32)
        return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

    raise ValueError(f"Unsupported pose file: {path}")

def load_beats(beats_path: Path, T: int) -> np.ndarray:
    """Load beats.json into length-T indicator in [0,1]."""
    if not beats_path.exists(): return np.zeros((T,), dtype=np.float32)
    b = json.loads(beats_path.read_text())
    if "beat_onehot" in b: x = np.array(b["beat_onehot"], dtype=np.float32)
    elif "onehot" in b:   x = np.array(b["onehot"], dtype=np.float32)
    else:
        x = np.array(b.get("beat_strength", [0]*T), dtype=np.float32)
        if x.max() > 1: x = x / (x.max() + 1e-6)
    if x.ndim == 0: x = np.full((T,), float(x), dtype=np.float32)
    if x.shape[0] < T:
        y = np.zeros((T,), dtype=np.float32); y[:x.shape[0]] = x; x = y
    elif x.shape[0] > T:
        x = x[:T]
    return x

def load_joint_names_from_meta(meta_path: str) -> list[str] | None:
    """Read ordered joint names from meta.json if available."""
    if not meta_path: return None
    p = Path(meta_path)
    if not p.exists(): return None
    try:
        meta = json.loads(p.read_text())
        for key in ("joint_names","joints","names","keypoints","kpts"):
            if key in meta and isinstance(meta[key], list):
                names = [str(x) for x in meta[key]]
                # apply alias mapping
                return [ALIASES.get(n, n) for n in names]
    except Exception:
        pass
    return None

# ========================= Edges & Mapping =========================

def build_edges(schema: str, joint_names: list[str] | None, joints: int) -> list[tuple[int,int]]:
    """Return edge list in INDEX space for the current data order."""
    schema = (schema or "blazepose").lower()
    spec = SCHEMAS.get(schema, SCHEMAS["blazepose"])
    names_ref = spec["names"]
    edges_by_name = spec["edges_by_name"]

    if joint_names and len(joint_names) == joints:
        idx = {n: i for i, n in enumerate(joint_names)}
        edges = [(idx[a], idx[b]) for a, b in edges_by_name if a in idx and b in idx]
        if edges: return edges

    # Fallback: assume dataset order == reference order (truncate if needed)
    idx_default = {n: i for i, n in enumerate(names_ref[:joints])}
    return [(idx_default[a], idx_default[b]) for a, b in edges_by_name
            if a in idx_default and b in idx_default
            and idx_default[a] < joints and idx_default[b] < joints]

def name_to_index_map(schema: str, joint_names: list[str] | None, joints: int) -> dict[str, int]:
    """Map joint name → index using meta names if valid, else schema reference order."""
    schema = (schema or "blazepose").lower()
    spec = SCHEMAS.get(schema, SCHEMAS["blazepose"])
    if joint_names and len(joint_names) == joints:
        return {n: i for i, n in enumerate(joint_names)}
    return {n: i for i, n in enumerate(spec["names"][:joints])}

# ========================= Bounds & Transforms =========================

def robust_bounds(P: np.ndarray, coords: int, q_low=1, q_high=99, fallback=(720,720)):
    """Percentile-clamped bounds from data; auto-detect normalized [0..1]."""
    D = P.shape[1]; J = D // coords
    XY = P.reshape(-1, J, coords)[..., :2].reshape(-1, 2)
    XY = XY[~np.isnan(XY).any(axis=1)]
    if XY.size == 0: return fallback[0], fallback[1], -1, 1, -1, 1, True
    x = XY[:,0]; y = XY[:,1]
    x0, x1 = np.percentile(x, [q_low, q_high])
    y0, y1 = np.percentile(y, [q_low, q_high])
    if (0 <= x0 <= 1 and 0 <= x1 <= 1 and 0 <= y0 <= 1 and 0 <= y1 <= 1):
        return fallback[0], fallback[1], 0, 1, 0, 1, True  # normalized
    pad_x = 0.05 * max(1e-6, x1 - x0); pad_y = 0.05 * max(1e-6, y1 - y0)
    W = int((x1 - x0 + 2*pad_x) or fallback[0])
    H = int((y1 - y0 + 2*pad_y) or fallback[1])
    return max(W, 320), max(H, 320), x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y, True

def world_to_img(points_xy, W, H, xmin, xmax, ymin, ymax, flip_y=True):
    x = points_xy[:,0]; y = points_xy[:,1]
    sx = (W - 1) / max(xmax - xmin, 1e-6)
    sy = (H - 1) / max(ymax - ymin, 1e-6)
    s = min(sx, sy)
    cx = (x - xmin) * s; cy = (y - ymin) * s
    if s == sx:
        pad = (H - 1) - (ymax - ymin) * s; cy = cy + pad * 0.5
    else:
        pad = (W - 1) - (xmax - xmin) * s; cx = cx + pad * 0.5
    if flip_y: cy = (H - 1) - cy
    return np.stack([np.clip(cx,0,W-1), np.clip(cy,0,H-1)], axis=-1).astype(np.int32)

def center_by_hips(P_arr: np.ndarray, joints: int, coords: int, name_idx: dict[str,int]) -> np.ndarray:
    """
    Subtract the hip midpoint (avg of left_hip/right_hip) from XY coords.
    Falls back to BlazePose indices 23/24 if names missing.
    """
    lh = name_idx.get("left_hip", 23)
    rh = name_idx.get("right_hip", 24)
    if P_arr.shape[1] != joints * coords: 
        return P_arr
    P3 = P_arr.reshape(-1, joints, coords)
    hips = 0.5 * (P3[:, lh, :2] + P3[:, rh, :2])  # [T,2]
    P3[:, :, :2] -= hips[:, None, :]
    return P3.reshape(P_arr.shape)

# ============================ Drawing ============================

def draw_skeleton(img, pts_xy, edges, nodes_color=(255,255,255), edges_color=(40,220,255), thickness=2):
    for a,b in edges:
        if 0 <= a < len(pts_xy) and 0 <= b < len(pts_xy):
            cv2.line(img, tuple(pts_xy[a]), tuple(pts_xy[b]), edges_color, thickness, cv2.LINE_AA)
    for p in pts_xy:
        cv2.circle(img, tuple(p), max(2, thickness-1), nodes_color, -1, lineType=cv2.LINE_AA)

# ============================= Render =============================

def render_video(
    poses_path: str,
    out_path: str,
    *,
    beats_path: str = "",
    joints: int = 33,
    coords: int = 2,
    fps: float = 30.0,
    size: str = "",
    background: str = "",
    no_flip_y: bool = False,
    debug: int = 0,
    meta_path: str = "",
    schema: str = "blazepose",
    codec: str = "mp4v",
    # NEW:
    magnify: float = 1.0,
    mag_ref: str = "first",   # "first" | "mean"
    center: bool = False,
):
    poses_path = Path(poses_path)
    out_path = Path(out_path)

    # load data
    P = load_pose_any(poses_path)  # [T, D]
    T, D = P.shape
    assert D == joints * coords, f"D={D} != joints*coords={joints*coords}"
    beats = load_beats(Path(beats_path), T) if beats_path else np.zeros((T,), dtype=np.float32)
    P = np.where(np.isfinite(P), P, 0.0)

    # joint names / indices
    joint_names = load_joint_names_from_meta(meta_path)
    name_idx = name_to_index_map(schema, joint_names, joints)

    # ---- Motion magnification (around reference) ----
    if mag_ref == "first":
        Pref = P[:1].copy()
    elif mag_ref == "mean":
        Pref = P.mean(axis=0, keepdims=True)
    else:
        Pref = P[:1].copy()

    if magnify != 1.0:
        P = Pref + magnify * (P - Pref)

    # ---- Optional hip-centering (on XY only) ----
    if center:
        P = center_by_hips(P, joints, coords, name_idx)

    # skeleton edges AFTER name mapping is established
    edges = build_edges(schema, joint_names, joints)

    # canvas / bounds (note: if --size is set, assumes normalized coords)
    if size:
        W, H = map(int, size.lower().split("x"))
        xmin=ymin=0.0; xmax=ymax=1.0; flip = not no_flip_y
    else:
        W, H, xmin, xmax, ymin, ymax, flip = robust_bounds(P, coords, fallback=(720,720))
        if no_flip_y: flip = False

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for '{out_path}' with codec '{codec}'")

    # optional background video
    bg_cap = None
    if background:
        bg_cap = cv2.VideoCapture(background)
        if not bg_cap.isOpened(): bg_cap = None

    # render
    J = joints; last_ok = None
    for t in range(T):
        # base frame
        if bg_cap:
            ok, frame = bg_cap.read()
            if not ok:
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = bg_cap.read()
            frame = cv2.resize(frame, (W, H))
            img = frame.copy()
        else:
            img = np.full((H, W, 3), 16, dtype=np.uint8)  # dark bg

        vec = P[t]
        if (not np.isfinite(vec).all()) or np.allclose(vec, 0):
            if last_ok is not None: vec = last_ok
        else:
            last_ok = vec

        xy = vec.reshape(J, coords)[:, :2]
        pts_img = world_to_img(xy, W, H, xmin, xmax, ymin, ymax, flip_y=flip)

        # border + skeleton + beat marker + time
        cv2.rectangle(img, (4,4), (W-5, H-5), (40,40,40), 1)
        draw_skeleton(img, pts_img, edges, thickness=2)
        cv2.circle(img, (20, 20), 8, (0,255,0) if beats[t] > 0.5 else (80,80,80), -1)
        cv2.putText(img, f"t={t+1}/{T}", (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

        if debug and t < debug:
            cv2.imwrite(str(out_path.with_name(out_path.stem + f"_dbg_{t:03d}.png")), img)

        vw.write(img)

    vw.release()
    if bg_cap: bg_cap.release()
    print(f"[saved] {out_path} ({T} frames @ {fps} fps, {W}x{H}, codec={codec})")

# ============================== CLI ==============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render pose sequences to video.")
    ap.add_argument("--poses", required=True, help="pred_poses.jsonl/json/npy/npz or GT poses")
    ap.add_argument("--out", required=True, help="output video, e.g. pred.mp4 / pred.avi")
    ap.add_argument("--beats", default="", help="optional beats.json to blink a marker")
    ap.add_argument("--meta", default="", help="optional meta.json with ordered joint names")
    ap.add_argument("--schema", default="blazepose", help="skeleton schema (default: blazepose)")
    ap.add_argument("--joints", type=int, default=33)
    ap.add_argument("--coords", type=int, default=2, choices=[2,3])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="", help="force WxH (normalized coords only), e.g. 720x720")
    ap.add_argument("--background", default="", help="optional background video path")
    ap.add_argument("--no_flip_y", action="store_true", help="do not invert Y (use image coords)")
    ap.add_argument("--debug", type=int, default=0, help="dump first N frames as PNGs")
    ap.add_argument("--codec", default="mp4v", help="fourcc (e.g. mp4v, avc1, MJPG)")
    # NEW args
    ap.add_argument("--magnify", type=float, default=1.0,
                    help="Scale motion around a reference (1.0=no change, try 50~200)")
    ap.add_argument("--mag_ref", choices=["first","mean"], default="first",
                    help="Reference for motion magnification")
    ap.add_argument("--center", action="store_true",
                    help="Center poses by subtracting hip center (left/right hip avg) before drawing")

    args = ap.parse_args()

    render_video(
        args.poses, args.out,
        beats_path=args.beats,
        joints=args.joints, coords=args.coords,
        fps=args.fps, size=args.size,
        background=args.background, no_flip_y=args.no_flip_y,
        debug=args.debug, meta_path=args.meta, schema=args.schema,
        codec=args.codec,
        magnify=args.magnify, mag_ref=args.mag_ref, center=args.center,
    )
