import re
from g2pk2 import G2p

_g2p = G2p()


def graphemes_to_phonemes_korean(raw_text: str) -> str:
    """
    raw_text: e. g. "한국어로 말하는 법을 배우는 중입니다."
    return:   "한구거로 말하는 버블 배우는 중임니다"
    """
    phonemes = _g2p(raw_text)
    phonemes = pron.strip()
    phonemes = re.sub(r"\s+", " ", phonemes)

    return phonemes
