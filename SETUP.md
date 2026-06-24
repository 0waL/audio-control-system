# 설치 및 실행 가이드

딥러닝 기반 다중 소음 분류 + 오디오 우선순위 제어 시스템.

> **핵심**: TensorFlow(YAMNet/CNN)는 **Python 3.10–3.13**에서만 동작한다.
> Python 3.14에는 아직 TensorFlow 휠이 없으므로, 딥러닝 경로는 **Python 3.12
> 가상환경**을 따로 만들어 사용한다.

---

## 1. 딥러닝 경로 (권장) — Python 3.12 가상환경

### 1) Python 3.12 설치
- <https://www.python.org/downloads/release/python-3120/> 에서 설치
- 설치 시 **"py launcher"** 옵션 체크 (이미 3.14가 있어도 공존 가능)

### 2) 가상환경 생성 · 활성화 (Windows)
```bat
cd C:\Users\wooji\audio-control-system
py -3.12 -m venv .venv
.venv\Scripts\activate
```
프롬프트 앞에 `(.venv)` 가 보이면 성공. `python --version` → `3.12.x` 확인.

### 3) 패키지 설치
```bat
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```
TensorFlow + YAMNet 포함. 최초 1회 실행 시 YAMNet 모델을 자동 다운로드한다.

### 4) 실행
```bat
python main.py --list-devices      # 마이크 장치 확인
python main.py --cli --verbose     # 헤드리스 실시간 (YAMNet)
python main.py                     # GUI 모드
```
로그에 다음이 보이면 딥러닝 분류가 동작 중이다:
```
YAMNet model loaded successfully
YAMNet→category mapping built: {'alarm': 19, 'speech': 12, 'music': 92, 'keyboard': 3, 'environment': 51}
```

특정 마이크를 쓰려면 인덱스를 지정한다 (예: MATA STUDIO C300 = 1):
```bat
python main.py --cli --device 1
```

---

## 2. 모델 직접 학습 (연구계획 2단계)

데이터셋을 받은 뒤 `training/train_cnn.py` 로 학습한다.

- **ESC-50**: <https://github.com/karoldvl/ESC-50> (`ESC-50-master` 압축 해제)
- **UrbanSound8K**: <https://urbansounddataset.weebly.com/urbansound8k.html>

```bat
:: YAMNet 임베딩 위에 분류 헤드만 학습 (가장 빠름, 전이학습)
python training\train_cnn.py --mode yamnet --esc50 path\to\ESC-50-master --urbansound path\to\UrbanSound8K

:: 멜-스펙트로그램 기반 CNN 직접 학습
python training\train_cnn.py --mode cnn --esc50 path\to\ESC-50-master --epochs 50

:: MFCC + RandomForest (가벼움, TF 불필요)
python training\train_cnn.py --mode mfcc --esc50 path\to\ESC-50-master

:: 평가 (정확도/정밀도/재현율/F1 + 혼동행렬)
python training\train_cnn.py --mode eval --model models\cnn_classifier.keras --esc50 path\to\ESC-50-master
```
학습된 모델은 `models/` 에 저장된다 (git 추적 제외). MFCC 모델은 실행 시
`python main.py --model models\mfcc_classifier.pkl` 로 불러올 수 있다.

---

## 3. 가벼운 경로 (선택) — Python 3.14 그대로 사용

TensorFlow 없이 규칙 기반(heuristic) / MFCC 분류만으로 빠르게 데모할 때:
```bat
pip install -r requirements.txt
python main.py --cli --no-yamnet
```
정확도는 낮지만 마이크 입력 → 분류 → 볼륨 제어 파이프라인 전체를 바로 확인할 수 있다.

---

## 참고
- 우선순위 설정은 `config/default_priorities.json` 이 기본값이며, 변경 시
  `~/.audio-control/priorities.json` 에 저장된다 (GUI에서 조정 가능).
- Windows 시스템 볼륨 제어는 `pycaw` 를 사용한다 (requirements에 자동 포함).
