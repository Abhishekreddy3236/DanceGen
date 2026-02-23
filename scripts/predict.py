# scripts/predict.py
# Generate pose sequences from a trained PoseTransformer.
# Supports teacher-forcing (tf), autoregressive (ar), and warmup (scheduled sampling) modes.
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch

from scripts.models.transformer_baseline import PoseTransformer
from scripts.models.train import _load_pose_any  # reuse robust JSONL/JSON/NPY loader

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_base_from_ckpt(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device)
    args = ck.get("args", {})
    joints = int(args.get("joints", 33))
    coords = int(args.get("coords", 2))
    beat_dim = int(args.get("beat_dim", 3))
    D = joints * coords
    base = PoseTransformer(in_dim=D + beat_dim, out_dim=D).to(device)
    state = ck.get("state", {}).get("base", ck.get("base"))
    base.load_state_dict(state, strict=False)
    base.eval()
    return base, joints, coords, beat_dim

def load_clip(clip_dir: str, joints: int, coords: int, beat_dim: int):
    """Load GT P [T,D] and beats X [T,beat_dim] from a ytXXX folder."""
    clip = Path(clip_dir)
    # poses file
    for name in ("poses.jsonl","poses.json","pose.json","poses.npy","poses.npz"):
        p = clip / name
        if p.exists(): poses_path = p; break
    else:
        raise FileNotFoundError(f"No poses file in {clip}")

    P = _load_pose_any(poses_path).astype(np.float32)  # [T, D]
    T, D = P.shape
    expD = joints * coords
    if D != expD:
        raise ValueError(f"Pose dim mismatch: file D={D} but expected {expD} for joints={joints}, coords={coords}")

    # beats
    beats_path = clip / "beats.json"
    if beats_path.exists():
        b = json.loads(beats_path.read_text())
        if "beat_onehot" in b:
            X = np.array(b["beat_onehot"], dtype=np.float32)
        elif "onehot" in b:
            X = np.array(b["onehot"], dtype=np.float32)
        else:
            X = np.array(b.get("beat_strength", [0]*T), dtype=np.float32)
            if X.max() > 1: X = X / (X.max() + 1e-6)
        if X.ndim == 1:
            # if scalar per frame, expand to beat_dim by repeating or zeros
            X = np.repeat(X[:, None], beat_dim, axis=1) if beat_dim > 1 else X[:, None]
    else:
        X = np.zeros((T, beat_dim), dtype=np.float32)

    # fit length
    if X.shape[0] < T:
        pad = np.zeros((T, X.shape[1]), dtype=np.float32); pad[:X.shape[0]] = X; X = pad
    elif X.shape[0] > T:
        X = X[:T]

    return P, X, poses_path

def hip_center(P: np.ndarray, joints: int, coords: int):
    """Subtract mid-hip (left/right) from XY to remove translation (BlazePose idx 23/24)."""
    if P.shape[1] != joints * coords: return P
    J = joints; C = coords
    P3 = P.reshape(-1, J, C)
    lh, rh = 23, 24
    if lh < J and rh < J:
        hips = 0.5 * (P3[:, lh, :2] + P3[:, rh, :2])  # [T,2]
        P3[:, :, :2] -= hips[:, None, :]
    return P3.reshape(P.shape)

def clamp_percentile(P: np.ndarray, q=0.5, Q=99.5):
    """Clamp to [q, Q] percentiles per feature to avoid outliers."""
    lo = np.nanpercentile(P, q, axis=0)
    hi = np.nanpercentile(P, Q, axis=0)
    return np.clip(P, lo, hi)

def write_jsonl(vecs: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in range(vecs.shape[0]):
            f.write(json.dumps(vecs[t].tolist()) + "\n")
    print(f"[saved] {out_path}  (T={vecs.shape[0]}, D={vecs.shape[1]})")

@torch.no_grad()
def generate_tf(base, P: torch.Tensor, Xbeat: torch.Tensor) -> np.ndarray:
    """Teacher forcing: feed previous GT pose."""
    P_prev = torch.cat([P[:, :1], P[:, :-1]], dim=1)  # [B,T,D]
    x_in = torch.cat([P_prev, Xbeat], dim=-1)         # [B,T,D+beat]
    Y = base(x_in)                                    # [B,T,D]
    return Y.squeeze(0).cpu().numpy()

@torch.no_grad()
def generate_ar(base, T: int, D: int, Xbeat: torch.Tensor, seed_frame: torch.Tensor):
    """Autoregressive: seed with 1 frame, then feed last prediction."""
    outs = []
    y_prev = seed_frame.clone()  # [1,1,D]
    for t in range(T):
        x_t = torch.cat([y_prev, Xbeat[:, t:t+1]], dim=-1)  # [1,1,D+beat]
        y_t = base(x_t)                                     # [1,1,D]
        outs.append(y_t)
        y_prev = y_t
    Y = torch.cat(outs, dim=1).squeeze(0).cpu().numpy()
    return Y

def _match_feat_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim columns so both arrays have the same feature dim."""
    d = min(a.shape[1], b.shape[1])
    return a[:, :d], b[:, :d]

@torch.no_grad()
def generate_warm(base, P: torch.Tensor, Xbeat: torch.Tensor, warmup: int = 30) -> np.ndarray:
    """
    First K frames with teacher forcing, then autoregressive for the rest.
    Returns exactly [T, D_use].
    """
    B, T, D = P.shape
    assert B == 1, "This implementation expects B=1"
    K = max(1, min(int(warmup), int(T)))

    # 1) TF on 0..K-1
    P_tf = P[:, :K]                # [1,K,D]
    X_tf = Xbeat[:, :K]            # [1,K,B]
    Y_tf = generate_tf(base, P_tf, X_tf)   # np [K, D_tf]

    # 2) AR on K..T-1 seeded by last TF frame
    seed = torch.from_numpy(Y_tf[-1:]).to(P.device).unsqueeze(0)  # [1,1,D_tf]
    outs = []
    for t in range(K, T):
        x_t = torch.cat([seed, Xbeat[:, t:t+1]], dim=-1)  # [1,1,D_seed+B]
        y_t = base(x_t)                                   # [1,1,D_ar]
        seed = y_t                                        # next seed
        outs.append(y_t)

    if outs:
        Y_ar = torch.cat(outs, dim=1).squeeze(0).cpu().numpy()  # [T-K, D_ar]
    else:
        Y_ar = np.zeros((0, Y_tf.shape[1]), dtype=Y_tf.dtype)    # edge case T==K

    # 3) Make feature dims match (trim to min)
    Y_tf, Y_ar = _match_feat_dims(Y_tf, Y_ar) if Y_ar.size else (Y_tf, Y_ar)

    # 4) Concat → [T, D_use]
    Y = np.concatenate([Y_tf, Y_ar], axis=0)
    return Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="runs/.../best.pt or ckpt_XXX.pt")
    ap.add_argument("--clip", required=True, help="clips like scripts/dataset_json/yt001")
    ap.add_argument("--out", default="", help="output jsonl (default: <clip>/pred_poses.jsonl)")
    ap.add_argument("--mode", choices=["tf","ar","warm"], default="tf",
                    help="tf=teacher forcing, ar=autoregressive, warm=warmup then AR")
    ap.add_argument("--warmup", type=int, default=30, help="frames of TF before AR (for mode=warm)")
    ap.add_argument("--motion_gain", type=float, default=1.0,
                    help="scale per-frame deltas around reference (>=1 amplifies motion)")
    ap.add_argument("--mag_ref", choices=["first","mean"], default="first",
                    help="reference for motion_gain")
    ap.add_argument("--center", action="store_true",
                    help="hip-center sequence (remove global XY translation)")
    ap.add_argument("--clip_percentile", type=float, default=0.0,
                    help="if >0, clamp to [p,100-p] per-dim percentiles (e.g. 0.5)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base, joints, coords, beat_dim = load_base_from_ckpt(args.ckpt, device)
    P_np, X_np, poses_path = load_clip(args.clip, joints, coords, beat_dim)  # [T,D], [T,B]
    T, D = P_np.shape

    # tensors
    P = torch.from_numpy(P_np).float().unsqueeze(0).to(device)       # [1,T,D]
    Xbeat = torch.from_numpy(X_np).float().unsqueeze(0).to(device)   # [1,T,B]

    # --- generation
    if args.mode == "tf":
        Y = generate_tf(base, P, Xbeat)  # [T,D]
    elif args.mode == "ar":
        seed = P[:, :1]                  # first GT frame as seed
        Y = generate_ar(base, T, D, Xbeat, seed)
    else:  # warm
        Y = generate_warm(base, P, Xbeat, warmup=args.warmup)

    # --- post-process (visualization-oriented)
    # 1) optional motion magnification (around a reference)
    if args.motion_gain != 1.0:
        if args.mag_ref == "first":
            ref = Y[:1].copy()
        else:
            ref = Y.mean(axis=0, keepdims=True)
        Y = ref + args.motion_gain * (Y - ref)

    # 2) optional centering by hips
    if args.center:
        Y = hip_center(Y, joints, coords)

    # 3) optional percentile clipping
    if args.clip_percentile > 0:
        p = float(args.clip_percentile)
        Y = clamp_percentile(Y, p, 100.0 - p)

    # write
    out_path = Path(args.out) if args.out else (poses_path.parent / "pred_poses.jsonl")
    write_jsonl(Y.astype(np.float32), out_path)

if __name__ == "__main__":
    main()