import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from g2pk2 import G2p
except Exception:
    G2p = None

# 🚨 중요: hangul_to_phoneme()는 반드시 ["ㅈ˟","ㅓ","ㄴ˳",...] 리스트를 생성하고
#          " ".join()까지 해서 최종 문자열을 return해야 합니다.
from matcha.text.korean_phoneme import hangul_to_phoneme


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
KSS_ROOT = DATA_ROOT / "kss"   # data/kss/{1,2,3,4}/**/*.wav
TRANSCRIPT = DATA_ROOT / "transcript.v.1.4.txt"
OUT_DIR = DATA_ROOT / "filelists"

_WS = re.compile(r"\s+")
_ALLOWED_HANGUL_LINE = re.compile(r"[^가-힣0-9A-Za-z\s\.\,\?\!\-~]")


# --------------------------
# utilities
# --------------------------
def _norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _clean_original(s: str) -> str:
    s = _ALLOWED_HANGUL_LINE.sub(" ", s)
    return _norm_ws(s)


def _read_transcript(path: Path) -> Dict[str, str]:
    """
    transcript.v.1.4.txt 한 줄 예:
      data/kss/4/4_2682.wav|한국어 텍스트|...|...
    → 항상 '첫 두 컬럼'만 취한다: (왼쪽key, 한국어 텍스트)
    """
    id2text: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue

            # 1st separator only (tab or pipe)
            if "\t" in ln:
                left, text = ln.split("\t", 1)
            elif "|" in ln:
                left, text = ln.split("|", 1)
            else:
                continue

            left, text = left.strip(), text.strip()

            # remove further columns
            if "|" in text:
                text = text.split("|", 1)[0].strip()
            if "\t" in text:
                text = text.split("\t", 1)[0].strip()

            key = Path(left).with_suffix("").name
            id2text[key] = text

    return id2text


def _scan_wavs() -> List[Path]:
    wavs: List[Path] = []
    for sub in ("1", "2", "3", "4"):
        wavs.extend((KSS_ROOT / sub).rglob("*.wav"))
    return sorted(wavs)


def _lookup_text(wav: Path, id2text: Dict[str, str]) -> str:
    stem = wav.stem
    cands = [stem, stem.upper(), stem.lower()]
    if "_" in stem:
        cands.append(stem.split("_")[-1])
    for c in cands:
        if c in id2text:
            return id2text[c]
    raise KeyError(f"Transcript not found for {wav}")


# --------------------------
# text routes
# --------------------------
def _to_syllable_g2pk2(s: str, g2p) -> str:
    s = g2p(s)
    s = _ALLOWED_HANGUL_LINE.sub(" ", s)
    return _norm_ws(s)


def _convert_text(route: str, original_text: str, g2p) -> str:
    base = _clean_original(original_text)

    if route == "original":
        return base

    elif route == "syllable":
        if g2p is None:
            raise RuntimeError("g2pk2 is required for syllable route.")
        return _to_syllable_g2pk2(base, g2p)

    elif route == "phoneme":
        # 🚨 hangul_to_phoneme() must return a space-separated unit string:
        # "ㅈ˟ ㅓ ㄴ˳ ㅡ ㄴ˳"
        return hangul_to_phoneme(base)

    else:
        raise ValueError(f"Unknown route: {route}")


# --------------------------
# building full filelists
# --------------------------
def _build_items(route: str, id2text: Dict[str, str]) -> List[str]:
    wavs = _scan_wavs()

    rel_paths: List[str] = []
    raw_texts: List[str] = []

    for wav in wavs:
        try:
            t = _lookup_text(wav, id2text)
        except KeyError:
            print(f"[warn] transcript not found for {wav}")
            continue

        rel_paths.append(wav.relative_to(ROOT).as_posix())
        raw_texts.append(_clean_original(t))

    items = []

    if route == "original":
        for rel, txt in zip(rel_paths, raw_texts):
            items.append(f"{rel}|{txt}")

    elif route == "syllable":
        if G2p is None:
            raise RuntimeError("g2pk2 is required for syllable route.")
        g2p = G2p()

        try:
            from tqdm import tqdm
            iterator = tqdm(raw_texts, desc="g2pk2")
        except Exception:
            iterator = raw_texts

        pron_texts = [_to_syllable_g2pk2(s, g2p) for s in iterator]
        for rel, txt in zip(rel_paths, pron_texts):
            items.append(f"{rel}|{txt}")

    elif route == "phoneme":
        try:
            from tqdm import tqdm
            iterator = tqdm(raw_texts, desc="phoneme")
        except Exception:
            iterator = raw_texts

        ph_texts = [hangul_to_phoneme(s) for s in iterator]
        for rel, txt in zip(rel_paths, ph_texts):
            # txt MUST be "token1 token2 token3 ..."
            items.append(f"{rel}|{txt}")

    else:
        raise ValueError(f"Unknown route: {route}")

    return items


# --------------------------
# split & write
# --------------------------
def _split_and_write(items: List[str], out_prefix: str, val_ratio: float, min_val: int) -> Tuple[Path, Path]:
    random.seed(42)
    random.shuffle(items)

    n_total = len(items)
    n_val = max(min_val, int(n_total * val_ratio))

    val_items = items[:n_val]
    train_items = items[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_p = OUT_DIR / f"{out_prefix}_train.txt"
    val_p = OUT_DIR / f"{out_prefix}_val.txt"

    train_p.write_text("\n".join(train_items), encoding="utf-8")
    val_p.write_text("\n".join(val_items), encoding="utf-8")

    return train_p, val_p


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Make KSS filelists (2 columns: wav|text).")

    ap.add_argument(
        "--route",
        choices=["original", "syllable", "phoneme"],
        default="original",
    )
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--min_val", type=int, default=100)
    ap.add_argument("--out_prefix", type=str, default=None)
    args = ap.parse_args()

    if not TRANSCRIPT.exists():
        raise FileNotFoundError(f"Transcript not found: {TRANSCRIPT}")

    id2text = _read_transcript(TRANSCRIPT)

    items = _build_items(args.route, id2text)

    prefix = args.out_prefix or f"kss_{args.route}"

    train_p, val_p = _split_and_write(items, prefix, args.val_ratio, args.min_val)

    print(f"[done:{args.route}] train={train_p}  val={val_p}  (total={len(items)})")


if __name__ == "__main__":
    main()
