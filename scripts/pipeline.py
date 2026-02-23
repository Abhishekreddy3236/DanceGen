#!/usr/bin/env python3
import csv, subprocess, sys, shutil
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# -------- Paths --------
ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"                # full downloads from yt-dlp
CLIPS = DATA / "clips"            # trimmed 30s clips
AUDIO = DATA / "audio"            # wav extracted from clips
MANIFEST = DATA / "manifest.csv"

for p in (RAW, CLIPS, AUDIO):
    p.mkdir(parents=True, exist_ok=True)

# -------- Config defaults (can be overridden per-row) --------
DEFAULT_TRIM_LEAD_S = 15          # set to 30 if you prefer that as your global default
DEFAULT_CLIP_LEN_S  = 30

# -------- Helpers --------
def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run(cmd: list[str], check=True):
    print(" $", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check)

@dataclass
class Row:
    video_id: str
    url: str
    label: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    trim_lead_s: Optional[float] = None
    clip_len_s: Optional[float] = None

    @staticmethod
    def from_dict(d: dict) -> "Row":
        def _f(x):
            x = (x or "").strip()
            return float(x) if x else None

        return Row(
            video_id = d.get("video_id","").strip(),
            url      = d.get("url","").strip(),
            label    = d.get("label","").strip(),
            start_s  = _f(d.get("start_s")),
            end_s    = _f(d.get("end_s")),    # not used for clipping; kept for compatibility
            trim_lead_s = _f(d.get("trim_lead_s")),
            clip_len_s  = _f(d.get("clip_len_s")),
        )

# -------- Core steps --------
def download_video(row: Row) -> Path:
    """
    Download the source video via yt-dlp into RAW as MP4.
    If it already exists, we skip.
    """
    out_mp4 = RAW / f"{row.video_id}.mp4"
    if out_mp4.exists():
        print(f"✓ Exists (skip download): {out_mp4.name}")
        return out_mp4

    if not have("yt-dlp"):
        sys.exit("ERROR: yt-dlp not found. Install: pip install yt-dlp")

    # We let yt-dlp choose best mp4 video+audio and merge to mp4
    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--write-info-json",
        "-o", str(RAW / f"{row.video_id}.%(ext)s"),
        row.url,
    ]
    print(f"↓ Downloading {row.video_id} ← {row.url}")
    run(cmd, check=True)
    if not out_mp4.exists():
        print(f"WARNING: {row.video_id}.mp4 missing after yt-dlp. Check output above.")
    return out_mp4

def make_clip(row: Row, src: Path) -> Path:
    """
    Create a 30s (default) clip after trimming the first 15s/30s (default) using ffmpeg.
    Output goes to CLIPS/{video_id}.mp4
    - We compute effective_start = (start_s or 0) + (trim_lead_s or DEFAULT_TRIM_LEAD_S)
    - We then take clip_len_s seconds from there (default DEFAULT_CLIP_LEN_S)
    """
    if not have("ffmpeg"):
        sys.exit("ERROR: ffmpeg not found. Install ffmpeg via your package manager.")

    clip = CLIPS / f"{row.video_id}.mp4"
    if clip.exists():
        print(f"✓ Exists (skip clip): {clip.name}")
        return clip

    trim_lead = row.trim_lead_s if row.trim_lead_s is not None else DEFAULT_TRIM_LEAD_S
    clip_len  = row.clip_len_s  if row.clip_len_s  is not None else DEFAULT_CLIP_LEN_S
    base_start = row.start_s if row.start_s is not None else 0.0
    effective_start = max(0.0, base_start + (trim_lead or 0.0))

    # Accurate seeking: put -ss AFTER -i, then -t for duration
    # + faststart for web playback + stream copy when possible; if copy fails, re-encode H.264/AAC
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-i", str(src),
        "-ss", f"{effective_start}",
        "-t",  f"{clip_len}",
        "-movflags", "+faststart",
        # try stream copy
        "-c", "copy",
        str(clip),
    ]
    print(f"✂️  Clipping {row.video_id}: start={effective_start}s len={clip_len}s → {clip.name}")
    r = run(cmd, check=False)
    if r.returncode != 0:
        # fallback re-encode for stubborn inputs
        print(" … stream copy failed; retrying with re-encode (libx264/aac)")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-i", str(src),
            "-ss", f"{effective_start}",
            "-t",  f"{clip_len}",
            "-movflags", "+faststart",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(clip),
        ]
        run(cmd, check=True)

    return clip

def extract_audio_from_clip(row: Row, clip: Path) -> Path:
    """
    Extract mono 22.05kHz WAV from the clipped video.
    """
    wav = AUDIO / f"{row.video_id}.wav"
    if wav.exists():
        print(f"✓ Exists (skip audio): {wav.name}")
        return wav

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-i", str(clip),
        "-vn",
        "-ac", "1",
        "-ar", "22050",
        str(wav),
    ]
    print(f"🎵 Extracting audio → {wav.name}")
    run(cmd, check=True)
    return wav

# -------- Orchestration --------
def process_manifest():
    if not MANIFEST.exists():
        sys.exit(f"ERROR: {MANIFEST} not found. Create it first.")

    with MANIFEST.open() as f:
        reader = csv.DictReader(f)
        required = {"video_id","url","label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: manifest missing required columns: {sorted(missing)}")

        rows = [Row.from_dict(d) for d in reader]

    for row in rows:
        if not row.video_id or not row.url:
            print(f"Skipping malformed row: {row}")
            continue
        print("\n=== JOB:", row.video_id, "===")
        src  = download_video(row)
        clip = make_clip(row, src)
        extract_audio_from_clip(row, clip)

    print("\nDone. Check:")
    print(f" - Raw downloads: {RAW}")
    print(f" - 30s clips:     {CLIPS}")
    print(f" - Audio WAVs:    {AUDIO}")

if __name__ == "__main__":
    process_manifest()