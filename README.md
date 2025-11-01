<div align="center">

# 🍵 Korean-Matcha-TTS: A fast Korean TTS architecture with conditional flow matching

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

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
python scripts/make_kss_filelists.py
# → data/filelists/kss_train.txt, kss_val.txt
```

`configs/data/kss.yaml`에 기재된 경로를 확인/수정합니다.

```yaml
train_filelist_path: data/filelists/kss_train.txt
valid_filelist_path: data/filelists/kss_val.txt
```

### E. Install espeak-ng

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
```

### F. Compute Mel Statistics

```bash
matcha-data-stats -i kss.yaml
# {'mel_mean': -6.562135219573975, 'mel_std': 2.7914016246795654}
```

결괏값을 `configs/data/kss.yaml`의 `data_statistics` 항목에 입력합니다.

```yaml
data_statistics:
  mel_mean: -6.562135
  mel_std:  2.791402
```

---

## 2️⃣ 학습

```bash
make train-kss
```
또는

```bash
python matcha/train.py experiment=kss
```

- 50 epochs 학습 후 종료, 5 epoch마다 체크포인트 저장
```bash
python matcha/train.py experiment=kss \
  trainer.max_epochs=50 \
  callbacks.model_checkpoint.every_n_epochs=5
```

- 50 epochs 학습 후 종료, 1000 step마다 체크포인트 저장
```bash
python matcha/train.py experiment=kss \
  trainer.max_epochs=50 \
  callbacks.model_checkpoint.every_n_train_steps=1000
```

- 최소 메모리 모드 (미지원)
  ```bash
  python matcha/train.py experiment=kss_min_memory
  ```
- 다중 GPU 학습  
  ```bash
  python matcha/train.py experiment=kss trainer.devices=[0,1]
  ```

---

## 3️⃣ 추론

Pre-trained **HiFi-GAN**에 의해 멜→오디오 변환이 수행됩니다.

```bash
matcha-tts   --text "한국어로 말하는 법을 배우고 있어요."   --checkpoint_path "<PATH_TO_CHECKPOINT>"   --vocoder hifigan_T2_v1   --steps 32   --out wavs/output
```

`steps`와 `temperature`는 trade-off 관계
- `--steps`: ODE 스텝 (값이 작을수록 빠르지만 노이즈↑)  
- `--temperature`: 샘플링 temperature

- download checkpoints
  - [1810 steps trained](https://drive.google.com/file/d/1eNhmS-AczUZCM9gc51DtaPQTWNy8QScm/view?usp=drive_link)
  - [3620 steps trained](https://drive.google.com/file/d/1sdjaAcckWl-0S6jcmGoFTKcZE9H-ye84/view?usp=drive_link)
  - [5430 steps trained](https://drive.google.com/file/d/1jpUqQAQTF9WXsJLYE-sgWs_peTgNz3rw/view?usp=drive_link)
  - [7240 steps trained](https://drive.google.com/file/d/11RGTlN_ZfWV1WnqlCz4TCBzkuT_lPGqc/view?usp=drive_link)
  - [9050 steps trained](https://drive.google.com/file/d/13sOrfpCLbCANbPnRL9spbgBrq9G8PU3S/view?usp=drive_link)
  - [10860 steps trained](https://drive.google.com/file/d/1onS0FACqc5x2VH3RDQUxsCYKOtwr_GMg/view?usp=drive_link)
  - [12670 steps trained](https://drive.google.com/file/d/1eYxfQHta6BNLyfdbxIIRbVfnLQIfFCia/view?usp=drive_link)
  - [14480 steps trained](https://drive.google.com/file/d/1Y5Rqidg227-1VGhXX9Y4fhoZsdE7oIPB/view?usp=drive_link)
  - [16290 steps trained](https://drive.google.com/file/d/1Lx7TfZoz9xD1abz7G7XJEcioagQYeA_n/view?usp=sharing)
  - [18100 steps trained](https://drive.google.com/file/d/1MLuDK7gfStKsJB7YhIS-i0uUE1ZI3YkI/view?usp=sharing)
---

## 4️⃣ 한국어 Matcha-TTS 개발 단계

### 1단계 — 한글(문자 단위) 기반 학습 (2025. 11. 1. ~)
- 한글/숫자/기본 문장부호만 남겨 텍스트 파일리스트에 기록  
- **G2P 없이도** 기본적인 음질 확보 가능  
- 연음·유음·경음화 등 발음규칙은 반영되지 않음
- 현재 `kss-without-phonemizer` 브랜치에서 관련 코드 개선 작업 중

### 2단계 — G2P 도입 (2025. 11. 2. ~)
- [g2pk2](https://pypi.org/project/g2pk2) 또는 [g2pkk](https://github.com/g2pkteam/g2pkk) 사용  
- 발음 규칙 반영(phonetise) 후 `text` 필드에 발음 표기 기록

---

## 5️⃣ ONNX Export

```bash
pip install onnx onnxruntime-gpu
python3 -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps 5
python3 -m matcha.onnx.infer model.onnx --text "안녕하세요" --gpu
```

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


