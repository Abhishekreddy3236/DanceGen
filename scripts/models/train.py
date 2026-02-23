# scripts/models/train.py
# Train DanceGen on a JSON dataset (baseline + optional diffusion).
from __future__ import annotations

import argparse, csv, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---- Relative imports to your models ----
from .transformer_baseline import PoseTransformer
from .diffusion_head import MotionLatent, EpsilonHead


# ======================= Config / Defaults =======================
L1_W, L2_W, L3_W, L4_W, L5_W = 1.0, 0.5, 0.1, 0.02, 0.05
BETA = 0.2  # diffusion weight

# BlazePose-33-ish bones (adjust if your order differs)
BLAZEPOSE_33_BONES = [
    (11,13),(13,15),
    (12,14),(14,16),
    (23,25),(25,27),
    (24,26),(26,28),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (15,17),(16,18),
    (27,29),(28,30),
    (29,31),(30,32),
]

def filter_bones(bones: List[Tuple[int,int]], joints: int) -> List[Tuple[int,int]]:
    out = [(a,b) for (a,b) in bones if a < joints and b < joints]
    if len(out) < len(bones):
        print(f"[warn] filtered {len(bones)-len(out)} bones out of range for joints={joints}")
    if not out:
        out = [(i, i+1) for i in range(max(0, joints-1))]
        print(f"[warn] no valid bones after filtering; using chain len={len(out)}")
    return out

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ======================= Metrics =======================
def mpjpe(pred: torch.Tensor, targ: torch.Tensor, coords: int) -> torch.Tensor:
    # pred,targ: [B,T,D] with D = J*coords
    B, T, D = pred.shape
    J = D // coords
    pred = pred.view(B, T, J, coords)
    targ = targ.view(B, T, J, coords)
    return ((pred - targ).pow(2).sum(-1).sqrt()).mean()

def pck(pred: torch.Tensor, targ: torch.Tensor, coords: int, thresh: float) -> torch.Tensor:
    B, T, D = pred.shape
    J = D // coords
    pred = pred.view(B, T, J, coords)
    targ = targ.view(B, T, J, coords)
    dist = (pred - targ).pow(2).sum(-1).sqrt()  # [B,T,J]
    return (dist < thresh).float().mean()


# ======================= Losses =======================
def losses(P_hat: torch.Tensor, P: torch.Tensor,
           bones_idx: List[Tuple[int,int]], coords_per_joint: int) -> torch.Tensor:
    B, T, D = P.shape
    assert D % coords_per_joint == 0, f"D={D} not divisible by coords={coords_per_joint}"

    # L1 position + velocity
    L_rec = (P_hat - P).abs().mean()
    Vh = P_hat[:,1:] - P_hat[:,:-1]
    V  = P[:,1:] - P[:,:-1]
    L_vel = (Vh - V).abs().mean()

    # Bone length variance over time
    def bone_len(X, a, b):
        a0, b0 = a*coords_per_joint, b*coords_per_joint
        return ((X[:,:,a0:a0+coords_per_joint] - X[:,:,b0:b0+coords_per_joint])**2).sum(-1).sqrt()  # [B,T]

    L_bone = 0.0
    for a,b in bones_idx:
        bl = bone_len(P_hat, a, b)
        L_bone = L_bone + bl.var(dim=1, unbiased=False).mean()

    # Smooth acceleration
    L_smooth = (Vh[:,1:] - Vh[:,:-1]).pow(2).mean() if T >= 3 else P_hat.new_tensor(0.0)

    # Placeholder foot-skate
    L_foot = P_hat.new_tensor(0.0)

    return L1_W*L_rec + L2_W*L_vel + L3_W*L_bone + L4_W*L_smooth + L5_W*L_foot


# ======================= Diffusion Schedule =======================
class SimpleNoiseSchedule:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        t = torch.linspace(0, 1, steps=num_steps, device=device)
        betas = beta_start + (beta_end - beta_start) * t
        alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)


# ======================= Forward Step =======================
def forward_step(batch: Dict[str, torch.Tensor],
                 nets: Dict[str, nn.Module], *,
                 with_diffusion: bool,
                 noise_sched: Optional[SimpleNoiseSchedule],
                 coords_per_joint: int,
                 bones_idx: List[Tuple[int,int]]) -> Tuple[torch.Tensor, Dict[str,float]]:
    P, Xbeat = batch["P"], batch["Xbeat"]   # [B,T,D], [B,T,beat_dim]
    B, T, D = P.shape

    P_prev = torch.cat([P[:, :1], P[:, :-1]], dim=1)
    x_in   = torch.cat([P_prev, Xbeat], dim=-1)

    Y = nets["base"](x_in)
    assert Y.shape == P.shape, f"pred {Y.shape} vs target {P.shape}"

    L_base = losses(Y, P, bones_idx=bones_idx, coords_per_joint=coords_per_joint)
    total = L_base
    logs = {"loss_base": float(L_base.item())}

    if with_diffusion:
        assert noise_sched is not None
        Z = nets["latent"].encode(P)  # [B,T,latent_dim] (adapt to your impl)
        t = torch.randint(0, noise_sched.num_steps, (B,), device=P.device)
        alpha = noise_sched.alpha_cumprod[t].view(B,1,1)
        eps = torch.randn_like(Z)
        Zt = alpha.sqrt()*Z + (1 - alpha).sqrt()*eps
        eps_hat = nets["eps_head"](Zt, t, Xbeat)
        L_diff = (eps_hat - eps).pow(2).mean()
        total = total + BETA*L_diff
        logs["loss_diff"] = float(L_diff.item())

    logs["loss_total"] = float(total.item())
    return total, logs


# ======================= JSON Dataset =======================
def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def _load_pose_any(path: Path) -> np.ndarray:
    """
    Load poses from:
      - poses.json / pose.json: list-of-lists or dict{'poses'|'P'|'data'|...}
      - poses.jsonl / pose.jsonl: one JSON object per line; accepts:
          * [f1, f2, ...]                      -> vector
          * {"pose": [...]} (or P/data/kpts/...) -> vector/2D -> flatten
          * [timestamp, [...]]                  -> take second element
      - poses.npy / poses.npz (npz: pick 'P'|'poses'|'pose'|'arr_0')
    Returns float32 array [T, D]. If D varies, pads to the max D with zeros.
    """
    def _to_vec(x) -> np.ndarray:
        """Return 1D float array from various shapes."""
        if isinstance(x, dict):
            for k in ("pose","P","data","keypoints","kpts","joints","coords"):
                if k in x:
                    x = x[k]
                    break
            else:
                # fallback: take values and flatten in key order
                x = [v for _, v in sorted(x.items())]
        a = np.array(x, dtype=np.float32)
        if a.ndim == 1:
            return a
        # if 2D (e.g., [J, C]) flatten
        return a.reshape(-1)

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for k in ("poses","P","data","keypoints","kpts","joints","coords"):
                if k in data:
                    arr = np.asarray(data[k], dtype=np.float32)
                    break
            else:
                arr = np.asarray(list(data.values()), dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            return arr.astype(np.float32)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    elif path.suffix == ".jsonl":
        rows: list[np.ndarray] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # case: [vec] or [t, vec]
                if isinstance(obj, (list, tuple)):
                    if len(obj) == 2 and isinstance(obj[1], (list, tuple, dict)):
                        vec = _to_vec(obj[1])
                    else:
                        vec = _to_vec(obj)
                else:
                    vec = _to_vec(obj)
                rows.append(vec)

        if not rows:
            return np.zeros((0, 0), dtype=np.float32)

        # pad to the maximum dimensionality
        D = max(r.shape[0] for r in rows)
        out = np.zeros((len(rows), D), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
        return out

    elif path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    elif path.suffix == ".npz":
        z = np.load(path)
        key = next((k for k in ("P","poses","pose","arr_0") if k in z.files), z.files[0])
        arr = z[key].astype(np.float32)
        if arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    else:
        raise ValueError(f"Unsupported pose file: {path}")

def _beats_to_tensor(beats_json: Dict, T: int) -> torch.Tensor:
    """
    Build Xbeat [T,3] from beats.json.
      onehot   ← 'beat_onehot' or 'onehot' (list/array or scalar→broadcast)
      strength ← 'beat_strength' or 'strength'
      tempo    ← 'tempo_perc' (preferred) or 'tempo' (array or scalar), else 'tempo_bpm' (scalar→/200)
    Missing values -> zeros/broadcast to T. Trim/pad to T as needed.
    """
    def arr_like(key_list, default=0.0):
        for k in key_list:
            if k in beats_json:
                v = beats_json[k]
                a = np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple, np.ndarray)) else np.array([float(v)], dtype=np.float32)
                return a
        return np.array([default], dtype=np.float32)

    onehot   = arr_like(["beat_onehot", "onehot"], 0.0)
    strength = arr_like(["beat_strength", "strength"], 0.0)

    if "tempo_perc" in beats_json:
        tempo = np.asarray(beats_json["tempo_perc"], dtype=np.float32)
    elif "tempo" in beats_json:
        tempo = np.asarray(beats_json["tempo"], dtype=np.float32)
    else:
        bpm = float(beats_json.get("tempo_bpm", 0.0))
        tempo = np.array([bpm/200.0], dtype=np.float32)

    # fit lengths
    def fit(x):
        if x.ndim == 0: x = np.array([float(x)], dtype=np.float32)
        if x.shape[0] == T: return x
        if x.shape[0] > T:  return x[:T]
        out = np.zeros((T,), dtype=np.float32); out[:x.shape[0]] = x; return out

    onehot = fit(onehot); strength = fit(strength); tempo = fit(tempo)
    X = np.stack([onehot, strength, tempo], axis=-1)  # [T,3]
    return torch.from_numpy(X).float()

def _fit_length(x: np.ndarray, T: int) -> np.ndarray:
    if x.shape[0] == T: return x
    if x.shape[0] > T:  return x[:T]
    out = np.zeros((T,), dtype=x.dtype)
    out[:x.shape[0]] = x
    return out

class JsonClipDataset(Dataset):
    """
    Accepts either:
      A) root/clipA/(poses.json|pose.json|poses.npy|poses.npz), root/clipB/...
      B) root/(poses.json|pose.json|poses.npy|poses.npz)  (root itself is a single clip)

    Each clip may also contain beats.json (optional).
    """
    POSE_CANDIDATES = ("poses.json", "pose.json", "poses.jsonl", "pose.jsonl", "poses.npy", "poses.npz")

    def __init__(self, root: str, beat_dim: int = 3):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"JSON dataset root not found: {self.root}")
        self.beat_dim = beat_dim

        def find_pose_file(d: Path) -> Optional[Path]:
            for name in self.POSE_CANDIDATES:
                f = d / name
                if f.exists():
                    return f
            return None

        # Case B: root itself is a clip
        clips: list[Path] = []
        if find_pose_file(self.root):
            clips = [self.root]
        else:
            # Case A: direct children
            for p in sorted(self.root.iterdir()):
                if p.is_dir() and find_pose_file(p):
                    clips.append(p)
            # fallback: recursive (one level or any depth)
            if not clips:
                for f in self.root.glob("**/poses.json"):
                    clips.append(f.parent)
                for f in self.root.glob("**/pose.json"):
                    clips.append(f.parent)
                # dedupe
                seen = set()
                clips = [c for c in clips if not (c in seen or seen.add(c))]

        if not clips:
            print("[debug] scan report of root subdirs/files:")
            if self.root.is_dir():
                for p in sorted(self.root.iterdir()):
                    if p.is_dir():
                        print("  dir:", p.name, "contains:",
                              [q.name for q in sorted(p.iterdir())[:10]], "...")
                    else:
                        print("  file:", p.name)
            raise FileNotFoundError(f"No clip folders with a pose file {self.POSE_CANDIDATES} under {self.root}")

        self.clips = clips
        print(f"[info] JSON dataset '{self.root}': {len(self.clips)} clips")
        # show first few for sanity
        for c in self.clips[:5]:
            pf = find_pose_file(c)
            print(f"  • {c.name}  (pose={pf.name if pf else '??'}, "
                  f"beats={'beats.json' if (c/'beats.json').exists() else 'missing'})")

    def __len__(self): 
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip = self.clips[idx]

        # load poses from any supported type
        pose_path = None
        for name in self.POSE_CANDIDATES:
            f = clip / name
            if f.exists():
                pose_path = f; break
        assert pose_path is not None, f"pose file vanished in {clip}"

        P = _load_pose_any(pose_path)          # [T, D], float32 np.array
        T, _ = P.shape

        beats_path = clip / "beats.json"
        if beats_path.exists():
            beats = json.loads(beats_path.read_text())
            X = _beats_to_tensor(beats, T)     # [T, 3]
        else:
            X = torch.zeros(T, self.beat_dim, dtype=torch.float32)

        return {"P": torch.from_numpy(P).float(), "Xbeat": X}


def make_loader_json(root: str, batch: int, shuffle=True, num_workers=0) -> DataLoader:
    ds = JsonClipDataset(root)
    def collate(items: List[Dict[str,torch.Tensor]]):
        T = max(x["P"].shape[0] for x in items)
        D = items[0]["P"].shape[1]
        Bdim = items[0]["Xbeat"].shape[1]
        P = torch.zeros(len(items), T, D)
        X = torch.zeros(len(items), T, Bdim)
        for i, it in enumerate(items):
            t = it["P"].shape[0]
            P[i,:t] = it["P"]; X[i,:t] = it["Xbeat"]
        return {"P": P, "Xbeat": X}
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate, num_workers=num_workers)


# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser(description="DanceGen training on JSON dataset")
    p.add_argument("--variant", default="baseline", choices=["baseline","with_diffusion"])
    p.add_argument("--json_data", required=True, help="Train JSON dataset root (folders with poses.json)")
    p.add_argument("--json_val", default="", help="Optional val JSON dataset root")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--coords", type=int, default=3, choices=[2,3], help="2 for (x,y), 3 for (x,y,z)")
    p.add_argument("--joints", type=int, default=33)
    p.add_argument("--beat_dim", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--pck_thresh", type=float, default=0.05, help="PCK threshold (same units as your data)")
    return p.parse_args()


# ======================= Main =======================
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    step_log_csv = out_dir / "train_log.csv"
    epoch_log_csv = out_dir / "epoch_log.csv"
    best_ckpt = out_dir / "best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dims & bones
    D = args.joints * args.coords
    in_dim, out_dim = D + args.beat_dim, D
    bones = filter_bones(BLAZEPOSE_33_BONES, args.joints)
    print(f"[info] joints={args.joints} coords={args.coords} D={D} bones={len(bones)}")

    # Models
    base = PoseTransformer(in_dim=in_dim, out_dim=out_dim).to(device)
    nets: Dict[str, nn.Module] = {"base": base}
    with_diffusion = (args.variant == "with_diffusion")
    noise_sched = None
    if with_diffusion:
        latent = MotionLatent(in_dim=out_dim, latent_dim=256).to(device)
        eps_head = EpsilonHead(latent_dim=256, cond_dim=args.beat_dim).to(device)
        nets.update({"latent": latent, "eps_head": eps_head})
        noise_sched = SimpleNoiseSchedule(num_steps=args.diffusion_steps, device=device)

    # Opt
    params = [p for m in nets.values() for p in m.parameters()]
    opt = optim.AdamW(params, lr=args.lr)

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if "state" in ckpt:
            for k in nets:
                nets[k].load_state_dict(ckpt["state"][k], strict=False)
        elif "base" in ckpt:
            nets["base"].load_state_dict(ckpt["base"], strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[info] resumed from {args.resume} @ epoch {start_epoch-1}")

    # Data: JSON roots
    train_loader = make_loader_json(args.json_data, args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader_json(args.json_val, args.batch, shuffle=False, num_workers=args.num_workers) \
                 if args.json_val else None

    # CSV headers
    if not step_log_csv.exists():
        with open(step_log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss"])
    if not epoch_log_csv.exists():
        with open(epoch_log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_mpjpe","val_pck"])

    best_score = float("inf")  # lower (val MPJPE) is better

    # Train
    for epoch in range(start_epoch, args.epochs + 1):
        base.train()
        if with_diffusion:
            nets["latent"].train(); nets["eps_head"].train()

        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k,v in batch.items()}
            loss, _ = forward_step(batch, nets,
                                   with_diffusion=with_diffusion, noise_sched=noise_sched,
                                   coords_per_joint=args.coords, bones_idx=bones)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(nets["base"].parameters()), 1.0)
            if with_diffusion:
                nn.utils.clip_grad_norm_(list(nets["latent"].parameters())+
                                         list(nets["eps_head"].parameters()), 1.0)
            opt.step()

            running += float(loss.item())
            if step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                print(f"epoch {epoch:03d} step {step:05d} loss={avg:.4f}")
                with open(step_log_csv, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, step, f"{avg:.6f}"])

        # epoch summary + evaluation
        train_epoch_loss = (running / max(step % args.log_every, 1)) if running != 0.0 else 0.0

        val_loss = val_mpj = val_pckv = None
        if val_loader is not None:
            base.eval()
            if with_diffusion:
                nets["latent"].eval(); nets["eps_head"].eval()
            tot_loss = tot_mpj = tot_pck = 0.0; n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vb = {k: v.to(device) for k,v in vb.items()}
                    # forward for prediction
                    P, Xb = vb["P"], vb["Xbeat"]
                    P_prev = torch.cat([P[:,:1], P[:,:-1]], dim=1)
                    x_in   = torch.cat([P_prev, Xb], dim=-1)
                    Y = base(x_in)
                    l = losses(Y, P, bones_idx=bones, coords_per_joint=args.coords)
                    tot_loss += float(l.item())
                    tot_mpj  += float(mpjpe(Y, P, args.coords).item())
                    tot_pck  += float(pck(Y, P, args.coords, args.pck_thresh).item())
                    n += 1
            val_loss = tot_loss / max(n,1)
            val_mpj  = tot_mpj  / max(n,1)
            val_pckv = tot_pck  / max(n,1)

        # Save checkpoints
        ckpt_path = out_dir / "checkpoints" / f"ckpt_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "args": vars(args),
            "state": {k: v.state_dict() for k, v in nets.items()},
            "train_epoch_loss": train_epoch_loss,
            "val_loss": val_loss,
            "val_mpjpe": val_mpj,
            "val_pck": val_pckv,
        }, ckpt_path)
        print(f"saved: {ckpt_path}")

        # Best by val MPJPE if available, else by train loss
        score = val_mpj if val_mpj is not None else train_epoch_loss
        if score is not None and score < best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "state": {k: v.state_dict() for k, v in nets.items()},
                "best_score": float(best_score),
                "metric": "val_mpjpe" if val_mpj is not None else "train_loss",
            }, best_ckpt)
            print(f"[best] updated {best_ckpt} (metric="
                  f"{'val_mpjpe' if val_mpj is not None else 'train_loss'} value={best_score:.4f})")

        # Epoch CSV
        with open(epoch_log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_epoch_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                "" if val_mpj  is None else f"{val_mpj:.6f}",
                "" if val_pckv is None else f"{val_pckv:.6f}",
            ])


if __name__ == "__main__":
    main()
