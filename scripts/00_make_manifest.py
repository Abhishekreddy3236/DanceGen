import csv, pathlib

# ---- Defaults (change these if you want) ----
DEFAULT_TRIM_LEAD_S = 15    # set to 30 if you want to always drop first 30s
DEFAULT_CLIP_LEN_S  = 30    # 30-second clip length

# Optional per-link overrides:
#   key can be a 0-based index into LINKS or the exact URL string.
#   Fields you can override: start_s, end_s, trim_lead_s, clip_len_s, label
OVERRIDES = {
    # examples:
    # 0: {"trim_lead_s": 30},  # for first link, trim 30s
    # "https://youtu.be/IdqAnUpceZs?si=2W4pbZIOkHD7YPL6": {"clip_len_s": 20},
}

LINKS = [
    ("https://youtu.be/zV1qLYukTH8?si=g25N0q9Ekpp1aKK4", "ballet"),
    ("https://youtu.be/gvLihpGqfXQ?si=d1a2Y85zOe47Qt1O", "ballet"),
    ("https://youtu.be/9lpR7yQ66mc?si=mLBAcAFygwAkZSX8", "ballet"),
    ("https://youtu.be/VhRPPeYbd4E?si=J7s20PxTCdXGMLby", "ballet"),
    ("https://youtu.be/BnJ5Akvsw6c?si=Do0csnX15CKEGkAO", "ballet"),
    ("https://youtu.be/IdqAnUpceZs?si=2W4pbZIOkHD7YPL6", "ballet"),
    ("https://youtu.be/50lAMbJUXfc?si=SxxKLjUqNQa6utqv", "ballet"),
    ("https://youtu.be/fzejb0djXAU?si=JSrs6HDUgry_Is1P", "ballet"),
    ("https://youtu.be/OykfoTv0zoY?si=5KYnkmeN5bjy--FJ", "ballet"),
    ("https://youtu.be/YX6lUW-_HiU?si=8j5oFPOiVrnw8lLf", "ballet"),
    ("https://youtu.be/NFZj6ZNd60s?si=CuEL3eomFKX1qDYI", "ballet"),
    ("https://youtu.be/vtYt7f5QIcE?si=8ccR-4U68oVt7y9E", "ballet"),
    ("https://youtu.be/kR6m7KNQAoo?si=YSWQgj5mHXFhcv1v", "ballet"),
    ("https://youtu.be/C172-vkz9RI?si=QxUGstscBiIe-9Sb", "ballet"),
    ("https://youtu.be/swWR8Ux1LWs?si=VH9Xkzeg8ar9APGD", "ballet"),
    ("https://youtu.be/DVMgNtUsQ5k?si=YtJoY2z4PThRo14d", "ballet"),
    ("https://youtu.be/b9WKL4QHqvg?si=i-9lHw4yeDE5SSWg", "ballet"),
    ("https://youtu.be/W0e15zMp1yk?si=hrkqCB3mOKDAWDqg", "ballet"),
]

out = pathlib.Path("data/manifest.csv")
out.parent.mkdir(parents=True, exist_ok=True)

fields = ["video_id","url","start_s","end_s","label","trim_lead_s","clip_len_s"]

def get_override(i, url):
    # try index-based, then url-based
    if i in OVERRIDES:
        return OVERRIDES[i]
    if url in OVERRIDES:
        return OVERRIDES[url]
    return {}

with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, (url, label) in enumerate(LINKS, start=1):
        vid = f"yt{i:03d}"
        ovr = get_override(i-1, url)

        row = {
            "video_id": vid,
            "url": url,
            "start_s": ovr.get("start_s", ""),   # keep blank unless you set
            "end_s":   ovr.get("end_s", ""),     # not used by clip step, kept for compatibility
            "label":   ovr.get("label", label),
            "trim_lead_s": ovr.get("trim_lead_s", DEFAULT_TRIM_LEAD_S),
            "clip_len_s":  ovr.get("clip_len_s",  DEFAULT_CLIP_LEN_S),
        }
        w.writerow(row)

print(f"Wrote {out}")
