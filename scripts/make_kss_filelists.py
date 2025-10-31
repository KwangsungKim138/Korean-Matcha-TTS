import sys, os, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
KSS_ROOT = DATA_ROOT / "kss"     # data/kss/{1,2,3,4}
TRANSCRIPT = DATA_ROOT / "transcript.v.1.4.txt"
OUT_DIR = DATA_ROOT / "filelists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- parse transcript.v.1.4.txt -----
# support pattern "wav_path|text" and pattern "id|text"
id2text = {}
with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        parts = None
        for sep in ["|", "\t"]:
            if sep in ln:
                parts = [p.strip() for p in ln.split(sep, 1)]
                break
        if not parts or len(parts) < 2:
            continue

        left, text = parts[0], parts[1]
        if left.lower().endswith(".wav"):
            key = Path(left).with_suffix("").name
        else:
            key = Path(left).name

        id2text[key] = text

# scan .wav
items = []
for sub in ["1", "2", "3", "4"]:
    for wav in (KSS_ROOT / sub).rglob("*.wav"):
        stem = wav.stem
        candidates = [stem, stem.upper(), stem.lower()]
        if "_" in stem:
            candidates.append(stem.split("_")[-1])

        text = None
        for c in candidates:
            if c in id2text:
                text = id2text[c]
                break
        if not text:
            print(f"[warn] transcript not found for {wav}")
            continue
        rel = wav.relative_to(ROOT)
        items.append(f"{rel.as_posix()}|{text}")

random.seed(42)
random.shuffle(items)

# train 90 / val 10 split
n_total = len(items)
n_val = max(100, int(n_total * 0.1))
val = items[:n_val]
train = items[n_val:]

with open(OUT_DIR / "kss_train.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(train))
with open(OUT_DIR / "kss_val.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(val))

print(f"[done] train: {len(train)}, val: {len(val)} → {OUT_DIR}")
