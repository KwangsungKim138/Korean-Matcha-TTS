# scripts/make_kss_filelists.py
import argparse
import random
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---- optional deps (install when using that route) ----
# pip install g2pk2 phonemizer
try:
    from g2pk2 import G2p  # for route=phoneme / ipa
except Exception:
    G2p = None

try:
    from phonemizer import phonemize  # for route=ipa
except Exception:
    phonemize = None

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
KSS_ROOT = DATA_ROOT / "kss"   # data/kss/{1,2,3,4}/**/*.wav
TRANSCRIPT = DATA_ROOT / "transcript.v.1.4.txt"
OUT_DIR = DATA_ROOT / "filelists"

_HANGUL = re.compile(r"[가-힣]")
_WS = re.compile(r"\s+")
_ALLOWED_HANGUL_LINE = re.compile(r"[^가-힣0-9A-Za-z\s\.\,\?\!\-~]")  # 기본 정리용
_ALLOWED_IPA = set(
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

def _norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()

def _read_transcript(path: Path) -> Dict[str, str]:
    """
    transcript.v.1.4.txt 한 줄 예:
      data/kss/4/4_2682.wav|박완서는 살아 있는 가장 뛰어난 한국 작가 중 한 사람이야.|...|...|5.5|...
    → 항상 '첫 두 컬럼'만 취한다: (왼쪽키, 한국어 텍스트)
    """
    id2text: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue

            # 탭/파이프 모두 지원하되, '최초 구분자 1회만' 분리
            # (왼쪽=ID 또는 wav경로, 오른쪽=한국어 텍스트)
            if "\t" in ln:
                left, text = ln.split("\t", 1)
            elif "|" in ln:
                left, text = ln.split("|", 1)   # ← maxsplit=1 이 핵심
            else:
                # 구분자 없으면 스킵
                continue

            left, text = left.strip(), text.strip()

            # 혹시 뒤쪽에 추가 |컬럼이 또 붙어있다면 버림
            if "|" in text:
                text = text.split("|", 1)[0].strip()
            if "\t" in text:
                text = text.split("\t", 1)[0].strip()

            # key는 wav stem 또는 id의 마지막 토큰(확장자 제거)
            key = Path(left).with_suffix("").name
            id2text[key] = text
    return id2text


def _scan_wavs() -> List[Path]:
    wavs: List[Path] = []
    for sub in ("1", "2", "3", "4"):
        wavs.extend((KSS_ROOT / sub).rglob("*.wav"))
    return sorted(wavs)

def _lookup_text(wav: Path, id2text: Dict[str, str]) -> str:
    stem = wav.stem  # e.g., "4_4069" or "4_4069_000"
    cands = [stem, stem.upper(), stem.lower()]
    if "_" in stem:
        cands.append(stem.split("_")[-1])  # 맨 뒤 토큰으로도 시도
    for c in cands:
        if c in id2text:
            return id2text[c]
    raise KeyError(f"Transcript not found for {wav}")

def _clean_grapheme(s: str) -> str:
    # 이모지/이상문자 제거 + 공백 정리
    s = _ALLOWED_HANGUL_LINE.sub(" ", s)
    return _norm_ws(s)

def _to_phoneme_g2pk2(s: str, g2p) -> str:
    # g2pk2로 한국어 발음 문자열 생성 (한글 표기)
    s = g2p(s)
    s = _ALLOWED_HANGUL_LINE.sub(" ", s)  # 한글/기본문장부호 위주만 남김
    return _norm_ws(s)


def _to_ipa_batch(lines, chunk_size=512, njobs=None, with_stress=False, progress=True):
    """
    lines: List[str] — 한국어 문장 리스트
    chunk_size: int — 한 번에 phonemize에 넘길 문장 수(256~1024 권장)
    njobs: int|None — 병렬 프로세스 수(기본: CPU 코어 수)
    with_stress: bool — 강세 기호 포함 여부
    progress: bool — tqdm 진행바 표시
    return: List[str] — IPA 변환 결과(허용되지 않는 기호는 공백 치환 후 공백 정규화)
    """
    if not lines:
        return []

    if njobs is None:
        import os
        njobs = max(1, (os.cpu_count() or 1))

    # 진행바 이터레이터
    it = range(0, len(lines), chunk_size)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=(len(lines) + chunk_size - 1) // chunk_size, desc="IPA")
        except Exception:
            pass  # tqdm 미설치 시 그냥 진행

    out = []
    for start in it:
        chunk = lines[start:start + chunk_size]
        ipa_list = phonemize(
            chunk,
            language="ko",
            backend="espeak",
            strip=True,
            with_stress=with_stress,
            njobs=njobs,
        )
        # 필터링 + 공백 정규화
        for ipa in ipa_list:
            cleaned = "".join(ch if ch in _ALLOWED_IPA else " " for ch in ipa)
            cleaned = _WS.sub(" ", cleaned).strip()
            out.append(cleaned)
    return out

def _to_ipa(text: str) -> str:
    # 단일 문장용 래퍼 (진행바/병렬 꺼서 오버헤드 최소화)
    return _to_ipa_batch([text], chunk_size=512, njobs=None, with_stress=False, progress=False)[0]


def _convert_text(route: str, text: str, g2p) -> str:
    # route별로 "모델이 먹을 최종 문자열"만 돌려준다.
    base = _clean_grapheme(text)
    if route == "grapheme":
        return base
    elif route == "phoneme":
        if g2p is None:
            raise RuntimeError("g2pk2 is required for route=phoneme. Please `pip install g2pk2`.")
        return _to_phoneme_g2pk2(base, g2p)
    elif route == "ipa":
        if g2p is None or phonemize is None:
            raise RuntimeError("g2pk2 and phonemizer are required for route=ipa. `pip install g2pk2 phonemizer`.")
        # 권장: g2pk2로 숫자/영문/연음 규칙 먼저 정규화 → IPA
        ph = _to_phoneme_g2pk2(base, g2p)
        return _to_ipa(ph)
    else:
        raise ValueError(f"Unknown route: {route}")

def _build_items(route: str, id2text: Dict[str, str]) -> List[str]:
    wavs: List[Path] = _scan_wavs()
    # 1) 원문 텍스트 한꺼번에 수집
    rel_paths: List[str] = []
    raw_texts: List[str] = []
    for wav in wavs:
        try:
            t = _lookup_text(wav, id2text)
        except KeyError:
            print(f"[warn] transcript not found for {wav}")
            continue
        rel_paths.append(wav.relative_to(ROOT).as_posix())
        raw_texts.append(_clean_grapheme(t))

    items: List[str] = []
    if route == "grapheme":
        # 2-a) 그래핌: 클린 텍스트 그대로
        for rel, txt in zip(rel_paths, raw_texts):
            items.append(f"{rel}|{txt}")

    elif route == "phoneme":
        # 2-b) g2pk2 일괄 처리
        if G2p is None:
            raise RuntimeError("g2pk2 is required for route=phoneme. Please `pip install g2pk2`.")
        g2p = G2p()
        # (원하면 tqdm)
        try:
            from tqdm import tqdm
            it = tqdm(raw_texts, desc="g2pk2")
        except Exception:
            it = raw_texts
        ph_texts = [_to_phoneme_g2pk2(s, g2p) for s in it]
        for rel, txt in zip(rel_paths, ph_texts):
            items.append(f"{rel}|{txt}")

    elif route == "ipa":
        # 2-c) g2pk2 → IPA 를 **배치**로
        if G2p is None or phonemize is None:
            raise RuntimeError("g2pk2 and phonemizer are required for route=ipa. `pip install g2pk2 phonemizer`.")
        g2p = G2p()
        try:
            from tqdm import tqdm
            it = tqdm(raw_texts, desc="g2pk2")
        except Exception:
            it = raw_texts
        ph_texts = [_to_phoneme_g2pk2(s, g2p) for s in it]

        # 여기가 핵심: 한 방에 phonemize
        ipa_texts = _to_ipa_batch(
            ph_texts,
            chunk_size=1024,   # 길면 1024, 보수적으로 512
            njobs=None,        # CPU 코어 수 자동
            with_stress=False,
            progress=True,     # tqdm 진행바
        )
        for rel, txt in zip(rel_paths, ipa_texts):
            items.append(f"{rel}|{txt}")

    else:
        raise ValueError(f"Unknown route: {route}")

    return items

def _split_and_write(items: List[str], out_prefix: str, val_ratio: float, min_val: int) -> Tuple[Path, Path]:
    random.seed(42)
    random.shuffle(items)
    n_total = len(items)
    n_val = max(min_val, int(n_total * val_ratio))
    val = items[:n_val]
    train = items[n_val:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_p = OUT_DIR / f"{out_prefix}_train.txt"
    val_p = OUT_DIR / f"{out_prefix}_val.txt"
    train_p.write_text("\n".join(train), encoding="utf-8")
    val_p.write_text("\n".join(val), encoding="utf-8")
    return train_p, val_p

def main():
    ap = argparse.ArgumentParser(description="Make KSS filelists per route (2 columns only).")
    ap.add_argument("--route", choices=["grapheme", "phoneme", "ipa"], default="grapheme")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--min_val", type=int, default=100)
    ap.add_argument("--out_prefix", type=str, default=None)
    # 선택: ipa 배치 파라미터
    ap.add_argument("--ipa_chunk", type=int, default=1024)
    ap.add_argument("--ipa_njobs", type=int, default=0, help="0=auto")
    args = ap.parse_args()

    if not TRANSCRIPT.exists():
        raise FileNotFoundError(f"Transcript not found: {TRANSCRIPT}")

    id2text = _read_transcript(TRANSCRIPT)

    # ipa 파라미터 주입 (간단히 오버라이드하려면 _to_ipa_batch의 기본값을 바꿔도 됨)
    if args.route == "ipa":
        # 간단히 래핑해서 기본값 갱신
        def _ipa_batch_with_args(lines):
            return _to_ipa_batch(
                lines,
                chunk_size=args.ipa_chunk,
                njobs=(None if args.ipa_njobs in (None, 0) else args.ipa_njobs),
                with_stress=False,
                progress=True,
            )
        globals()["_to_ipa_batch"] = _ipa_batch_with_args  # 주입 (간단 해킹)
    items = _build_items(args.route, id2text)

    prefix = args.out_prefix or f"kss_{args.route}"
    train_p, val_p = _split_and_write(items, prefix, args.val_ratio, args.min_val)
    print(f"[done:{args.route}] train={train_p}  val={val_p}  (total={len(items)})")


if __name__ == "__main__":
    main()
