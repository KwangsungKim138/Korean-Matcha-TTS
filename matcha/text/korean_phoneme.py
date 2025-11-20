import re
from g2pk2 import G2p

_CHOSEONG  = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
_JUNGSEONG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
_JONGSEONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

_MARK_INIT = "˟"  # 어절 첫소리 태그
_MARK_CODA = "˳"  # 받침 태그
_MARK_PAL  = "ʲ"  # 구개음화

_EOJEOL_INIT = {"ㄱ","ㄷ","ㅂ","ㅈ"}
_PALATAL = {"ㅅ","ㅆ","ㄴ","ㄹ","ㅎ"}
_Y_VOWELS = {"ㅣ","ㅑ","ㅕ","ㅛ","ㅠ","ㅖ","ㅒ","ㅟ"}

_WS = re.compile(r"\s+")
_HANGUL_AND_PUNC = re.compile(r"[^가-힣0-9A-Za-z\s\.\,\?\!\-~]")

_g2p = G2p()


def _split_syllable_with_tags(syllable: str, eojeol_first: bool):
    """한 글자를 음소 단위(자모+태그) 리스트로 분해."""
    code = ord(syllable)
    if not (0xAC00 <= code <= 0xD7A3):
        return [syllable]

    base = code - 0xAC00
    cho = _CHOSEONG[base // 588]
    jung = _JUNGSEONG[(base % 588) // 28]
    jong = _JONGSEONG[base % 28]

    units = []

    # 1) 초성 태그
    if eojeol_first and cho in _EOJEOL_INIT:
        units.append(cho + _MARK_INIT)
    elif cho in _PALATAL and jung in _Y_VOWELS:
        units.append(cho + _MARK_PAL)
    else:
        units.append(cho)

    # 2) 중성
    units.append(jung)

    # 3) 종성 태그
    if jong:
        units.append(jong + _MARK_CODA)

    return units


def hangul_to_phoneme(text: str) -> str:
    """
    최종 출력: '토큰1 토큰2 토큰3 ...'
    예: 'ㅈ˟ ㅓ ㄴ˳ ㅡ ㄴ˳'
    """
    # 1) 발음 변환 (한글 음절)
    t = _g2p(text)
    t = _HANGUL_AND_PUNC.sub(" ", t)
    t = _WS.sub(" ", t).strip()

    units = []
    eojeol_first = True

    # 2) 각 음절 → 음소 단위 유닛 리스트 생성
    for ch in t:
        if ch == " ":
            eojeol_first = True
            continue

        units.extend(_split_syllable_with_tags(ch, eojeol_first))
        eojeol_first = False

    # 3) 토큰 간 공백으로 join → 모델이 읽을 수 있는 phoneme 스트림
    return " ".join(units)
