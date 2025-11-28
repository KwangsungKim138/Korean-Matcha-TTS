<div align="center">

# 🍵 Korean-Matcha-TTS: A fast Korean TTS architecture with conditional flow matching

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

</div>

> This repository builds upon [Matcha-TTS (ICASSP 2024)](https://arxiv.org/abs/2309.03199),
> initially trained on the [KSS dataset](https://huggingface.co/datasets/Bingsu/KSS_Dataset), 
> and designed to support additional Korean speech corpora in the future.
>  
> [🍵 Matcha-TTS (ICASSP 2024)](https://arxiv.org/abs/2309.03199)는 [Conditional Flow Matching](https://arxiv.org/abs/2210.02747)을 이용해 만든 
> 빠르고 자연스러운 **비자기회귀(Non-autoregressive)** 음성 합성 모델입니다.
> **Korean-Matcha-TTS**는 이를 기반으로 **KSS 한국어 음성 데이터셋**을 학습한 버전입니다.
---

## 🧠 특징

- 확률적·비자기회귀적 구조  
- 메모리 효율적
- 자연스러운 음질  
- ODE 기반의 빠른 합성  
- 한국어 KSS 데이터셋 대응

---

## ⚙️ 환경
 
- **Python:** 3.10  
- **PyTorch:** ≥ 2.0  

---

## 🧪 전처리 방식
 
한국어 TTS 모델을 학습할 때, 입력 텍스트를 어떤 단위로 표현하느냐에 따라 학습 효율과 음성 품질이 달라질 수 있습니다. 한글은 겉보기에는 ‘한 글자 = 한 음절’이지만, 실제로는 음소(초성·중성·종성) 조합으로 이루어져 있고, 음운 변동으로 인해 표기와 실제 발음이 서로 다른 경우가 많습니다.

따라서 다음 세 가지 방식으로 텍스트를 전처리하여 TTS 모델을 학습시키고, 각 방식이 모델 성능에 어떤 차이를 만드는지 실험했습니다.

### 1. 원문 문자 단위 (`--route original`)

* 한글을 문자 단위 그대로 모델에 입력
* 예: "한국말" → ["한", "국", "말"]
* 실제 발음과 다른 표기가 그대로 입력되므로 음운 변동을 모델이 스스로 학습해야 함

### 2. 음절 단위 (`--route syllable`)

* g2p 등을 적용하여 한국어 음운 변동 규칙을 반영, 텍스트를 음절 단위로 모델에 입력
* 예: "한국말" → "한궁말" → ["한", "궁", "말"]
* 현대 한글로 조합 가능한 글자 수는 11,172자로, 데이터셋이 모든 조합을 포함하고 있어야 한국어 음절을 모두 학습 가능

### 3. 음소 단위 (`--route phoneme`)

* 한 음절을 초성/중성/종성으로 분해하여 음소 단위로 학습
* 예: "한국말" → "한궁말" → ["ㅎ", "ㅏ", "ㄴ", "ㄱ", "ㅜ", "ㅇ", "ㅁ", "ㅏ", "ㄹ"]
* 표기는 같지만 발음이 다른 다음 경우에 대해서도 구분 처리
  * 어절 첫소리 평음 ㄱ, ㄷ, ㅂ, ㅈ은 무성음으로 발음되므로, 유성음 ㄱ, ㄷ, ㅂ, ㅈ과 구별
  * ㄱ, ㅇ 등 음향적 차이가 있는 초성과 종성도 별개의 단위로 구별
  * ㄴ, ㄹ, ㅅ, ㅆ, ㅎ 등의 자음은 'i', 'y' 계열의 모음 앞에서 구개음이 되므로 보통의 ㄴ, ㄹ, ㅅ, ㅆ, ㅎ와 구별
---

## 1️⃣ 설치

### A. Python / PyTorch

```bash
conda create -n korean-matcha-tts python=3.10 -y
conda activate korean-matcha-tts

# Install PyTorch CUDA 12.1 build
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### B. Clone Repository

```bash
git clone https://github.com/KwangsungKim138/Korean-Matcha-TTS.git
cd Korean-Matcha-TTS
pip install -e .
```

### C. Download & Unzip KSS Dataset

[KSS 데이터셋](https://huggingface.co/datasets/Bingsu/KSS_Dataset)을 다운로드한 뒤 data/kss 경로에서 압축을 해제합니다.

```
data/kss/
├─ 1/
│  ├─ 1_0000.wav
│  └─ ...
├─ 2/
├─ 3/
├─ 4/
└─ transcript.v.1.4.txt
```

### D. Make KSS Filelists

Matcha-TTS는 **Tacotron 2 스타일**(`path|text`) 파일리스트를 사용합니다.

```bash
python scripts/make_kss_filelists.py --route phoneme
# → --route < original | syllable | phoneme >
# → data/filelists/kss_phoneme_train.txt, kss_phoneme_val.txt
```

### E. Install espeak-ng

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
```

### F. Compute Mel Statistics

```bash
matcha-data-stats -i kss_phoneme.yaml
# → kss_< original | syllable | phoneme >.yaml
# {'mel_mean': -6.562135219573975, 'mel_std': 2.7914016246795654}
```

결괏값을 `configs/data/kss_phoneme.yaml`의 `data_statistics` 항목에 입력합니다.

```yaml
data_statistics:
  mel_mean: -6.562135
  mel_std:  2.791402
```

---

## 2️⃣ 학습

```bash
make train-kss
# make train-kss_original
# make train-kss_syllable
# make train-kss_phoneme
```
또는

```bash
python matcha/train.py experiment=kss_phoneme
# python matcha/train.py experiment=kss_original
# python matcha/train.py experiment=kss_syllable
```

- 50 epochs 학습 후 종료, 5 epoch마다 체크포인트 저장
```bash
python matcha/train.py experiment=kss_phoneme \
  trainer.max_epochs=50 \
  callbacks.model_checkpoint.every_n_epochs=5
```

- 50 epochs 학습 후 종료, 1000 step마다 체크포인트 저장
```bash
python matcha/train.py experiment=kss_phoneme \
  trainer.max_epochs=50 \
  callbacks.model_checkpoint.every_n_train_steps=1000
```
---

## 3️⃣ 추론

Pre-trained **HiFi-GAN**에 의해 멜→오디오 변환이 수행됩니다.

```bash
matcha-tts   --text "한국어로 말하는 법을 배우고 있어요."   --checkpoint_path "<PATH_TO_CHECKPOINT>"   --vocoder hifigan_T2_v1   --route <original | syllable | phoneme>   --steps 32   --out wavs/output
```

`steps`와 `temperature`는 trade-off 관계
- `--steps`: ODE 스텝 (값이 작을수록 빠르지만 노이즈↑)  
- `--temperature`: 샘플링 temperature

- download checkpoints
  - [문자 단위 50000 steps, '--route original'](https://drive.google.com/file/d/1HEOsPkewc7EPF6SPXUOWOji7CFW2qcFu/view?usp=sharing)
  - [음절 단위 50000 steps, '--route syllable'](https://drive.google.com/file/d/1_PqX4f9jCob6O7HdSRi7LNo5nL6RivJB/view?usp=sharing)
  - [음소 단위 50000 steps, '--route phoneme'](https://drive.google.com/file/d/1V_ynmXWU6WgJUS_jK00FOLPYrpXTciwL/view?usp=sharing)
 

### CER
| 학습 단위                        | sentence 1 | sentence 2 |
| :--------------------------- | :--------------: | :--------------: |
| 문자 단위 학습                     |      0.4286      |      0.4545      |
| 음절 단위 학습                     |      0.1071      |      0.2727      |
| 확장 음소 단위 학습 (Phoneme+Marker) |      0.0000      |      0.0455      |
---

## 📄 Citation

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{'e}kely, {'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024},
  note={This repository builds upon Matcha-TTS (ICASSP 2024) and adapts it for the Korean KSS dataset.}
}
```

---

- [Matcha-TTS (Original Repo)](https://github.com/shivammehta25/Matcha-TTS)  
