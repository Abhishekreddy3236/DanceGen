# scripts/eval.py
# Evaluate/visualize logs produced by scripts.models.train
import argparse, csv
from pathlib import Path

import matplotlib.pyplot as plt

def read_epoch_log(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            # coerce to float when possible
            for k in ("epoch","train_loss","val_loss","val_mpjpe","val_pck"):
                if k in row and row[k] != "":
                    row[k] = float(row[k]) if k != "epoch" else int(float(row[k]))
                else:
                    row[k] = None if k != "epoch" else None
            rows.append(row)
    return rows

def read_step_log(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["epoch"] = int(float(row["epoch"]))
            row["step"] = int(float(row["step"]))
            row["train_loss"] = float(row["train_loss"])
            rows.append(row)
    return rows

def plot_loss_curves(run_dir, epochs):
    xs = [e["epoch"] for e in epochs]
    tr = [e["train_loss"] for e in epochs]
    vl = [e["val_loss"] for e in epochs] if any(e["val_loss"] is not None for e in epochs) else None

    plt.figure()
    plt.plot(xs, tr, label="train loss")
    if vl is not None:
        plt.plot(xs, vl, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    out = Path(run_dir) / "loss_curves.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[saved] {out}")

def plot_val_metrics(run_dir, epochs):
    # only if we have val metrics
    if not any(e["val_mpjpe"] is not None or e["val_pck"] is not None for e in epochs):
        return
    xs = [e["epoch"] for e in epochs]
    mp = [e["val_mpjpe"] for e in epochs] if any(e["val_mpjpe"] is not None for e in epochs) else None
    pk = [e["val_pck"] for e in epochs] if any(e["val_pck"] is not None for e in epochs) else None

    plt.figure()
    if mp is not None:
        plt.plot(xs, mp, label="val MPJPE (↓)")
    if pk is not None:
        plt.plot(xs, pk, label="val PCK (↑)")
    plt.xlabel("epoch")
    plt.title("Validation metrics")
    plt.legend()
    out = Path(run_dir) / "val_metrics.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[saved] {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., runs/baseline_json_xy")
    ap.add_argument("--check_best", action="store_true", help="Load best.pt and print a shape sanity-check")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    epoch_log = run_dir / "epoch_log.csv"
    step_log  = run_dir / "train_log.csv"
    best_ckpt = run_dir / "best.pt"

    if not epoch_log.exists():
        raise FileNotFoundError(f"Missing {epoch_log}")

    epochs = read_epoch_log(epoch_log)
    print(f"[info] epochs logged: {len(epochs)}")
    if step_log.exists():
        steps = read_step_log(step_log)
        print(f"[info] step logs: {len(steps)} entries (averaged every N steps)")

    # text summary
    last = epochs[-1]
    tr_min = min(e["train_loss"] for e in epochs if e["train_loss"] is not None)
    tr_last = last["train_loss"]
    print(f"[summary] train_loss: last={tr_last:.4f}, min={tr_min:.4f}")

    if any(e["val_loss"] is not None for e in epochs):
        vl_vals = [e["val_loss"] for e in epochs if e["val_loss"] is not None]
        print(f"[summary] val_loss: last={vl_vals[-1]:.4f}, min={min(vl_vals):.4f}")
    if any(e["val_mpjpe"] is not None for e in epochs):
        vm_vals = [e["val_mpjpe"] for e in epochs if e["val_mpjpe"] is not None]
        print(f"[summary] val_mpjpe (↓): last={vm_vals[-1]:.4f}, min={min(vm_vals):.4f}")
    if any(e["val_pck"] is not None for e in epochs):
        vp_vals = [e["val_pck"] for e in epochs if e["val_pck"] is not None]
        print(f"[summary] val_pck (↑): last={vp_vals[-1]:.4f}, max={max(vp_vals):.4f}")

    # plots
    plot_loss_curves(run_dir, epochs)
    plot_val_metrics(run_dir, epochs)

    if args.check_best:
        try:
            import torch
            ck = torch.load(best_ckpt, map_location="cpu")
            args_in_ck = ck.get("args", {})
            metric = ck.get("metric", "train_loss")
            best = ck.get("best_score", None)
            print(f"[best] {best_ckpt.name} metric={metric} value={best}")
            for k in ("joints","coords","beat_dim"):
                if k in args_in_ck:
                    print(f"  {k} = {args_in_ck[k]}")
            # print model keys present
            state = ck.get("state", {})
            print(f"  components: {list(state.keys())}")
        except Exception as e:
            print(f"[warn] could not inspect best.pt: {e}")

if __name__ == "__main__":
    main()