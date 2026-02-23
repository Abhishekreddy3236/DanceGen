# features/build_features.py
import numpy as np, librosa, json, pathlib
from glob import glob

POSES = pathlib.Path("data/poses")
AUDIO = pathlib.Path("data/audio")
OUT   = pathlib.Path("data/features"); OUT.mkdir(parents=True, exist_ok=True)

FPS = 25

def norm_pose(P):
    # P: [T,33,2] in px → pelvis-centered, torso-scaled
    LHIP, RHIP, LSH, RSH = 23,24,11,12
    pelvis = (P[:,LHIP] + P[:,RHIP]) / 2
    P = P - pelvis[:,None,:]
    torso = np.linalg.norm(P[:,LSH] - P[:,RHIP], axis=1) + 1e-6
    P = P / torso[:,None,None]
    return P

def beat_feats(wav_path, T):
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    oenv  = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    beat_env = librosa.util.normalize(librosa.clicks(frames=beats, sr=sr, length=len(y)))
    # frame-align to video T frames
    hop = len(y)/T
    idx = (np.arange(T)*hop).astype(int)
    idx[idx>=len(y)] = len(y)-1
    strength = np.abs(beat_env[idx])
    onehot = np.zeros(T); onehot[(np.rint(librosa.frames_to_time(beats, sr=sr)*(FPS))).astype(int)%T] = 1.0
    return np.stack([onehot, strength, np.full(T, tempo/200.0)], axis=-1) # [T,3]

def run_one(stem):
    npz = np.load(POSES/f"{stem}.npz", allow_pickle=True)
    P = npz["kpts"]        # [T,33,2]
    P = norm_pose(P)
    T = P.shape[0]
    xbeat = beat_feats(AUDIO/f"{stem}.wav", T)   # [T,3]
    np.savez_compressed(OUT/f"{stem}.npz", P=P.astype(np.float32), Xbeat=xbeat.astype(np.float32), fps=FPS)

def main():
    stems = [pathlib.Path(p).stem for p in glob(str(POSES/"*.npz"))]
    for s in stems:
        run_one(s)
if __name__ == "__main__":
    main()
