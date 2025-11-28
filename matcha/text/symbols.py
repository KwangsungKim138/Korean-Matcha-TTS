"""from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
_hangul = [chr(c) for c in range(0xAC00, 0xD7A4)]
_digits = list("0123456789")

_cho = [
    "ㄱ˟","ㄱ","ㄲ","ㄴ","ㄴʲ","ㄷ˟","ㄷ","ㄸ","ㄹ","ㄹʲ","ㅁ","ㅂ˟","ㅂ","ㅃ",
    "ㅅ","ㅅʲ","ㅆ","ㅆʲ","ㅇ","ㅈ˟","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ","ㅎʲ"
]

_jung = [
    "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ",
    "ㅗ","ㅘ","ㅙ","ㅚ","ㅛ",
    "ㅜ","ㅝ","ㅞ","ㅟ","ㅠ",
    "ㅡ","ㅢ","ㅣ"
]

_jong = [
    "ㄱ˳","ㄴ˳","ㄷ˳","ㄹ˳",
    "ㅁ˳","ㅂ˳","ㅇ˳"
]

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_hangul) + list(_digits) + list(_cho) + list(_jung) + list(_jong)

# Special symbol ids
SPACE_ID = symbols.index(" ")

print(f"[symbols] size={len(symbols)}", flush=True)
