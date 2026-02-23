#!/usr/bin/env python3
import json, pathlib
from typing import Dict, Any
import numpy as np
import librosa

POSES_DIR = pathlib.Path("data/poses")        # contains ytXXX.npz (kpts, conf, fps, H, W, joints)
AUDIO_DIR = pathlib.Path("data/audio")        # contains ytXXX.wav
OUT_DIR   = pathlib.Path("dataset_json")      # new output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Helpers --------------------
def to_float_list(arr: np.ndarray):
    """Convert numpy array to nested python lists of float."""
    return np.asarray(arr).tolist()

def beat_features_aligned(wav_path: pathlib.Path, T: int, fps: float):
    """
    Returns a dict with per-frame beat features aligned to T video frames.
    - beat_onehot[t]
    - beat_strength[t] (onset energy proxy)
    - tempo_perc[t] (tempo normalized, optional)
    """
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    # onset envelope + beat tracking
    oenv  = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    # create an audio-length click signal at beats as a proxy, then sample per frame
    clicks = librosa.clicks(frames=beat_frames, sr=sr, length=len(y))
    # map video frames to audio indices
    # number of audio samples per video frame:
    hop = len(y) / T
    idx = (np.arange(T) * hop).astype(int)
    idx[idx >= len(y)] = len(y) - 1

    beat_strength = np.abs(clicks[idx])  # 0 or click amp around beats
    # onehot: mark frames that are nearest to a beat
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_video_frames = np.rint(beat_times * fps).astype(int)
    beat_video_frames = beat_video_frames[beat_video_frames < T]
    beat_onehot = np.zeros(T, dtype=np.float32)
    beat_onehot[beat_video_frames] = 1.0

    # simple tempo feature (scaled)
    tempo_perc = np.full(T, float(tempo) / 200.0, dtype=np.float32)

    return {
        "beat_onehot": to_float_list(beat_onehot),
        "beat_strength": to_float_list(beat_strength.astype(np.float32)),
        "tempo_perc": to_float_list(tempo_perc),
        "tempo_bpm": float(tempo),
    }

def write_json(path: pathlib.Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def write_jsonl_rows(path: pathlib.Path, rows: list[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

# -------------------- Main conversion per video --------------------
def process_stem(stem: str):
    pose_npz = POSES_DIR / f"{stem}.npz"
    wav_path = AUDIO_DIR / f"{stem}.wav"
    if not pose_npz.exists():
        print(f"[skip] {stem}: missing poses {pose_npz}")
        return
    if not wav_path.exists():
        print(f"[skip] {stem}: missing audio {wav_path}")
        return

    npz = np.load(pose_npz, allow_pickle=True)
    kpts: np.ndarray = npz["kpts"]      # [T,33,2] pixel coords (already smoothed/interpolated)
    conf: np.ndarray = npz["conf"]      # [T,33]
    fps: float = float(npz["fps"])
    H = int(npz["H"]); W = int(npz["W"])
    joints = list(npz["joints"]) if "joints" in npz else [f"j{i}" for i in range(kpts.shape[1])]
    T = kpts.shape[0]

    # ---- beats aligned to frames ----
    beats = beat_features_aligned(wav_path, T=T, fps=fps)

    # ---- write per-video folder ----
    out_dir = OUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) meta.json
    meta = {
        "video_id": stem,
        "fps": fps,
        "num_frames": T,
        "frame_size": {"H": H, "W": W},
        "num_joints": int(kpts.shape[1]),
        "joints": joints,
        "files": {
            "poses": "poses.jsonl",
            "conf": "conf.jsonl",
            "beats": "beats.json",
        }
    }
    write_json(out_dir / "meta.json", meta)

    # 2) beats.json
    write_json(out_dir / "beats.json", beats)

    # 3) poses.jsonl
    pose_rows = []
    for t in range(T):
        pose_rows.append({
            "t": int(t),
            "kpts": to_float_list(kpts[t])  # [[x,y], ...] len=33
        })
    write_jsonl_rows(out_dir / "poses.jsonl", pose_rows)

    # 4) conf.jsonl
    conf_rows = []
    for t in range(T):
        conf_rows.append({
            "t": int(t),
            "conf": to_float_list(conf[t])  # [c0, c1, ...]
        })
    write_jsonl_rows(out_dir / "conf.jsonl", conf_rows)

    print(f"[ok] {stem} → {out_dir}")

def main():
    stems = [p.stem for p in POSES_DIR.glob("*.npz")]
    if not stems:
        print("No pose npz files found in data/poses/")
        return
    for s in sorted(stems):
        process_stem(s)

if __name__ == "__main__":
    main()