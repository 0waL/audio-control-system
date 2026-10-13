# YAMNet on UrbanSound8K

- Date: 2026-10-12
- Dataset: `archive (1).zip`; 8732 evaluated files, 0 read/inference errors
- Model: TF Hub google/yamnet/1 with this project's fixed 521-to-5 class mapping; no trained head
- Audio: mono, 16 kHz; non-overlapping 1-second windows; final remainder included if at least 0.5 seconds
- File prediction: highest mean category probability over its windows; no temporal smoothing
- Ground truth: UrbanSound8K.csv mapped through `URBANSOUND_MAP`
- UrbanSound8K contains no speech or keyboard labels, so those categories are not measured for recall

## Overall

| Metric | Result |
| --- | --- |
| File accuracy | 5765/8732 = 66.02% |
| Window accuracy | 19602/31651 = 61.93% |
| Window detection (maximum category probability >= 0.3) | 27303/31651 = 86.26% |
| Macro F1 (represented labels only) | 0.746 |

## Accuracy by UrbanSound8K class

| Source class | Mapped category | Correct | Files | Accuracy |
| --- | --- | --- | --- | --- |
| air_conditioner | environment | 825 | 1000 | 82.5% |
| car_horn | alarm | 155 | 429 | 36.1% |
| children_playing | environment | 133 | 1000 | 13.3% |
| dog_bark | environment | 248 | 1000 | 24.8% |
| drilling | environment | 823 | 1000 | 82.3% |
| engine_idling | environment | 961 | 1000 | 96.1% |
| gun_shot | alarm | 290 | 374 | 77.5% |
| jackhammer | environment | 936 | 1000 | 93.6% |
| siren | alarm | 699 | 929 | 75.2% |
| street_music | music | 695 | 1000 | 69.5% |

## Confusion matrix (rows: true, columns: predicted)

| True / predicted | alarm | speech | music | keyboard | environment | unknown |
| --- | --- | --- | --- | --- | --- | --- |
| alarm | 1144 | 44 | 79 | 0 | 364 | 101 |
| speech | 0 | 0 | 0 | 0 | 0 | 0 |
| music | 3 | 227 | 695 | 0 | 73 | 2 |
| keyboard | 0 | 0 | 0 | 0 | 0 | 0 |
| environment | 134 | 1200 | 151 | 2 | 3926 | 587 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 |
