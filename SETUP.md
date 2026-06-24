# 설치 및 실행 가이드

딥러닝(PyTorch CNN) 기반 다중 소음 분류 + 오디오 우선순위 제어 시스템.

> **좋은 소식**: 딥러닝 백엔드를 **PyTorch**로 구현해서 **Python 3.14에서 그대로
> 동작**한다. TensorFlow처럼 별도 파이썬을 설치할 필요가 없다.

---

## 1. 설치 (Python 3.14 그대로)

```bat
cd C:\Users\wooji\audio-control-system
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```
- `requirements-ml.txt` = 코어 패키지 + PyTorch(CPU).
- PyTorch가 느리게 받아지거나 엉뚱한 빌드가 깔리면 CPU 전용 인덱스를 쓴다:
  ```bat
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- 딥러닝 없이 가볍게만 쓰려면 `pip install -r requirements.txt` 만 해도 된다
  (규칙기반/MFCC 분류로 동작).

---

## 2. 실행

```bat
python main.py --list-devices            # 마이크 장치 확인
python main.py --cli --verbose --device 1  # 헤드리스 실시간 (마이크 인덱스 지정)
python main.py                           # GUI 모드
```
- 학습된 CNN(`models/cnn_torch.pt`)이 있으면 자동으로 그걸 사용하고,
  없으면 규칙기반 분류로 폴백한다(정확도는 낮음).
- CNN을 끄고 규칙기반만 쓰려면 `--no-cnn`.

---

## 3. 모델 직접 학습 (연구계획 2단계)

데이터셋을 받은 뒤 `training/train_cnn.py` 로 학습한다.

- **ESC-50**: <https://github.com/karoldvl/ESC-50> (`ESC-50-master` 압축 해제)
- **UrbanSound8K**: <https://urbansounddataset.weebly.com/urbansound8k.html>

```bat
:: PyTorch CNN (멜-스펙트로그램 기반, 프로젝트의 딥러닝 모델)
python training\train_cnn.py --mode cnn --esc50 path\to\ESC-50-master --urbansound path\to\UrbanSound8K --epochs 50

:: MFCC + RandomForest (가벼움, GPU 불필요)
python training\train_cnn.py --mode mfcc --esc50 path\to\ESC-50-master

:: 평가 (정확도/정밀도/재현율/F1 + 혼동행렬)
python training\train_cnn.py --mode eval --model models\cnn_torch.pt --esc50 path\to\ESC-50-master
```
- 학습된 모델은 `models/` 에 저장된다(git 추적 제외).
- CNN: `models\cnn_torch.pt` → 실행 시 자동 로드(또는 `--cnn-model` 로 지정).
- MFCC: `models\mfcc_classifier.pkl` → `python main.py --model models\mfcc_classifier.pkl`.

> 데이터셋 없이 동작만 확인하려면 경로 인자 없이 실행하면 합성 데이터로
> 스모크 학습을 돌린다(정확도 의미 없음, 파이프라인 점검용).

---

## 참고
- 우선순위 설정은 `config/default_priorities.json` 이 기본값이며, 변경 시
  `~/.audio-control/priorities.json` 에 저장된다(GUI에서 조정 가능).
- Windows 시스템 볼륨 제어는 `pycaw` 를 사용한다(requirements에 자동 포함).
- 마이크 인덱스는 `python main.py --list-devices` 로 확인한다.
