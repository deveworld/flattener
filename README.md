# flattener

문서 사진의 곡면·원근 보정을 위한 Python 라이브러리입니다.

현재 상태: **실험용 · CUDA/XPU 학습 검증 · 실사진 품질 개선 중**.

현재 구현의 재현 검사, 모델 실패 원인과 개선 순서는 [통합 리뷰](docs/IMPLEMENTATION_REVIEW.md)에 정리했습니다.

## 설치

Linux x86_64 / Python 3.12 기준입니다. `uv.lock`은 CPU 전용 PyTorch를 고정합니다.

```bash
uv sync --locked --python 3.12
uv run --locked pytest -q
uv run --locked ruff check .
```

기본 테스트는 학습·역전파를 건너뜁니다. `--run-training`은 CPU 학습을 포함하는 명시적 허용 옵션입니다.

## 렌더링

```python
from flattener import identity_map, load_image, render, save_result

image = load_image("page.jpg")
grid = identity_map(image.shape[:2])
result = render(image, grid)
save_result(result, "outputs/page")
```

항등 좌표 예제입니다. 보정에는 출력→원본의 정규화 좌표 `float32 (H, W, 2)`를 전달합니다. 마지막 축은 `(x, y)`, 크기는 `(height, width)`입니다. API·마스크·실패 처리는 [인터페이스](docs/INTERFACE_SPEC.md)에 정리했습니다.

## 모델 추론

```python
from flattener import RenderConfig
from flattener.learning import unwarp
from flattener.learning.coordinate_study import load_coordinate_predictor

predictor = load_coordinate_predictor("outputs/coordinate-fold/selected.pt")
result = unwarp(image, predictor, (1024, 768), config=RenderConfig(on_error="return"))
save_result(result, "outputs/predicted-page")
```

신뢰할 수 있는 체크포인트와 출력 크기를 지정합니다. `quality_status="not_evaluated"`이며 글자 잘림과 잔여 휨을 확인해야 합니다. 파일 실행: `examples/unwarp_photo.py --help` (성공 종료 0, 실패 1).

## 좌표 모델 학습

장치는 `--device cpu`, `--device cuda`, `--device xpu`로 지정합니다. 기본값은 CUDA이며 다른 장치로 자동 전환하지 않습니다. Intel GPU에는 `torch.xpu.is_available()`이 참인 XPU용 PyTorch 환경을 사용합니다. XPU 환경에서는 해당 Python을 직접 실행합니다. 저장소의 `uv sync --locked`는 CPU용 의존성을 복원합니다.

```bash
python -m flattener.learning.coordinate_study check-device --device cuda
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m flattener.learning.coordinate_study train \
  outputs/dataset outputs/coordinate-next \
  --device cuda --architecture surface --epochs 40 --stop-after-epoch 12 --batch-size 32 --seed 20261003 \
  --input-size 384 288 --grid-size 9 7 --learning-rate 0.0001 \
  --topology-weight 100 --boundary-weight 0.5 --differential-weight 0.2 \
  --topology-mode relative --map-mode analytic --initialization-mode resolution \
  --background-probability 0.5 --rotation-probability 0.15 --rotation-mode per_image_fit \
  --gradient-clip-norm 5 --cache-mib 12288 --max-seconds 1200 \
  --persistent-cache outputs/dataset/preprocessed-hires.pt \
  --include-paper-augmentation --initial-checkpoint outputs/coordinate-fold/surface-e40-best.pt
```

CUDA용 비교 실험 설정입니다. 빈 출력 경로와 캐시 12GiB 예산을 지정합니다. 같은 모델의 전체 가중치를 이어받아 입력 크기만 바꾸고 optimizer를 초기화합니다. 12epoch 뒤 검증한 후, 같은 코드·환경·설정에서 `--initial-checkpoint`와 `--stop-after-epoch`를 빼고 `--resume OUTPUT/last.pt`로 재개합니다. 출력 디렉터리 전체가 필요하며 `--max-seconds`는 호출별 제한입니다.

전처리는 데이터 폴더의 `.pt`·JSON 캐시로 재사용합니다. 용량은 입력 192×144에서 4.48GB, 384×288에서 10.93GB입니다. 원본·전처리 코드·설정·환경이 바뀌면 `--rebuild-cache`가 필요합니다. 실행 경계에서는 전체 해시, epoch 사이에는 파일 상태와 코드 해시를 확인합니다. 연속 캐시에서 pinned 배치를 모아 GPU로 비동기 전송하며, `prepare_coordinate_data()` 결과를 여러 실행에 공유할 수 있습니다.

학습 증강: `--background-probability`는 보호된 종이 영역 밖의 배경을 절차적 질감으로 바꿉니다. `--rotation-mode per_image_fit`은 이미지별 90°·180°·270° 회전에 원본 종횡비를 유지하고 정답 좌표·마스크를 함께 변환합니다. 두 확률의 기본값은 0이고 검증 자료는 유지됩니다. `--page-relative-weight`와 `--orientation-weight`로 작은 문서·방향 손실을 조절할 수 있습니다.

신규 `--architecture oriented_surface --map-mode analytic`은 4방향 분류·페이지 마스크·실제 페이지 모서리를 함께 감독합니다. `--page-weight`, `--direction-weight`, `--corner-weight`를 모두 양수로 지정하며 회전 증강에는 `--rotation-mode per_image_fit`을 사용합니다. 새 출력층은 초기 상태에서 시작합니다. `backbone` 초기화는 같은 ModelConfig의 global 모델에서 encoder·공유 hidden 가중치만 가져옵니다. 학습 중 방향 혼동 행렬·모서리 오차·출처별 프레임/문서 평균을 `training.json`에 기록합니다. 페이지 crop·내부 보정 분리는 후속 작업입니다.

`--direction-training teacher`는 학습 중 증강 후 정답 방향으로 기하 손실을 계산합니다. 방향 분류는 별도 손실로 학습하며, 검증·추론은 예측 방향을 사용합니다. 기본값은 `predicted`입니다.

모서리·방향 정답은 메타데이터에서 계산해 이미지 캐시에 붙입니다. 캐시는 PyTorch 버전을 포함한 환경을 검사합니다. 체크포인트 0.12는 방향 학습 방식과 XPU RNG를 저장하며, 0.1~0.11 추론을 지원합니다.

RTX 3060 Ti에서 전처리 최초 생성은 11분 47초, 최적화 후 재사용은 1분 45초였습니다. 동일 모델·정밀도의 최종 반복 측정에서 배치 32는 35.0→32.0ms, 배치 128은 85.9→65.5ms, GPU 동기화는 배치당 54→9회였습니다. 전체 5,952장의 배치 준비는 1.52→0.72초입니다. 짧은 GPU 훈련·검증·저장은 epoch당 평균 8.3초였으며 실행 시작·종료의 전체 해시 검사는 별도입니다. 측정 조건·정확성 검증은 [performance.json](outputs/coordinate-fold/performance.json), 재현 자료는 `performance.zip`에 보관합니다.

## 최근 모델 결과

`ssh home`의 RTX 3060 Ti에서 공식 UVDoc를 개발 1,440장에 적용했습니다. 고정한 두 경로의 총 2,880회 CUDA 추론·좌표 저장은 32.16초, 입력 검사 포함 49.75초였습니다. 로컬 좌표 합성은 3.96초, 기하 검사는 34.68초, 정답 평가는 212.69초였습니다.

| 개발 자료 | 기준 좌표 오차 | UVDoc 직접 보정 | 기존 보정 뒤 UVDoc |
| --- | ---: | ---: | ---: |
| 자체 합성 768장 | 7.52px | 4.00px | 12.79px |
| 사진 기반 휨 합성 504장 | 6.90px | 6.08px | 12.06px |
| 촬영 168장 | 9.82px | 11.50px | 13.20px |

직접 보정은 기존 예측 crop·방향을 사용합니다. 자체 합성의 휨 차분 오차는 0.934→0.688px로 줄었고, 사진 기반 합성은 0.296→0.302px로 증가했습니다. 촬영 정답은 수작업 평면 근사입니다. 기존 보정 뒤 재적용한 경로는 페이지 보존율이 74~82%로 떨어졌습니다.

**기본 `selected.pt`를 유지합니다.** 두 경로 모두 교체 기준에 미달했습니다. 새 접힘·렌더링 실패는 0건이며, 기하 안전장치는 모든 보정량을 유지했습니다. 테스트 43개와 Ruff를 통과했고, 각 경로의 1,440장 원본 정답·기준 지표가 과거 평가와 정확히 일치했습니다. [평가 결과](outputs/coordinate-fold/uvdoc-evaluation.json)·[코드·기록](outputs/coordinate-fold/uvdoc-evaluation.zip).

UVDoc는 별도 Paddle 환경을 사용합니다. cuDNN 9.9.0.52를 고정했고, 배포 패키지의 의존성 선언과 충돌하는 항목을 기록했습니다. 기존 PyTorch 환경은 유지했습니다. [CUDA 입출력 검증](outputs/coordinate-fold/uvdoc.json).

앞선 4,512장·20epoch 잔차 학습은 4분 34초, 최대 할당 메모리 3.00GiB였습니다. [학습](outputs/coordinate-fold/feasible-cuda.json)·[평가](outputs/coordinate-fold/feasible-evaluation.json)·[손실 비교](outputs/coordinate-fold/tail-cuda.json)·[입력 비교](outputs/coordinate-fold/geometry-refiner.json)·[정답 직접 적합](outputs/coordinate-fold/residual.json) 기록을 보존합니다.

기준 좌표는 기존 방향·영역 추정 파이프라인의 출력입니다. 좌표 오차는 작업 이미지 384×288, 휨 차분은 17×13 좌표 격자의 정답 대비 2차 차분 기준입니다. 개발 문서는 잔차 모델 학습에서 제외했으며 앞선 모델들은 해당 문서를 학습했습니다. 추가 validation·test, 독립 실사진·OCR·SOTA 평가는 남아 있습니다.

## 데이터와 예제

데이터는 `outputs/dataset` 한곳에 보관합니다. `manifest.json`에 경로·분할·출처·해시를 기록하고 원본 파일은 `files/`에 둡니다.

SmartDoc의 실제 촬영 템플릿 30종·사진 360장에서 자료 2,400장을 만들었습니다. 원문별로 학습 22종·검증 4종·시험 4종을 나누고 원통형 휨·원근·크기·조명·흐림·노이즈·JPEG 변화를 적용했습니다. 관련 원문이 묶인 학습 source group은 14개입니다. 촬영 원본의 기하 정답은 유지합니다.

좌표 샘플은 기존 합성과 합쳐 학습 5,952·검증 528·시험 528장입니다. `--include-paper-augmentation`으로 새 자료를 포함합니다. DocUNet 사진 130장·스캔 65장은 평가용입니다. 보관 중인 SyntheticDoc 1,400개는 좌표 변환과 원문 재분할이 필요합니다. 출처·이용 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.

- `examples/generate_paired_scenes.py OUTPUT`: 자체 합성 장면 생성.
- `examples/unwarp_photo.py --help`: 사진 한 장 보정.
- `examples/evaluate_docunet.py --help`: DocUNet 빠른 점검(비표준). `--ocr`은 Tesseract 영문 인식 결과와 스캔 인식 결과의 불일치를 측정합니다. 논문 수치와의 비교는 [표준 벤치마크 평가](#표준-벤치마크-평가)를 사용합니다.

고정 모델 평가 명령입니다. 검증에는 `--split validation`을 사용합니다. 촬영·증강·기존 합성의 오차, 실패, 페이지 영역을 따로 기록합니다.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --locked python -m flattener.learning.evaluation \
  --dataset outputs/dataset --checkpoint outputs/coordinate-fold/selected.pt \
  --split test --output outputs/evaluation --output-size 192 144 --include-paper-augmentation
```

품질 기반 후보 선택은 `flattener.learning.evaluation.select_scene_checkpoints()`로 수행합니다. 기준 모델과 후보마다 `SceneSelectionCandidate(id, checkpoint_path, checkpoint_sha256, report_path, report_sha256)`를 전달합니다. 정식 validation 보고서의 기존 교체 조건을 검사하고 `best_eligible` 또는 `null`을 반환합니다. CLI는 `python -m flattener.learning.evaluation select --help`에서 확인합니다. `best.pt`는 평균 좌표 오차 기준을 유지하며, 선택 기능은 모델 파일을 교체하지 않습니다.

실패 배경의 원래 큰 회전 사례는 학습 자료의 2.2%입니다. 해당 페이지는 192×144 입력에서 중앙값 약 58×100px이며, 원본 해상도와 학습 증강을 추가해 비교했습니다. 방향·크기·위치 오차를 출처별로 기록합니다.

## 표준 벤치마크 평가

`flattener.benchmark`는 DocUNet의 MS-SSIM·LD·AD를 MATLAB 스크립트와 같은 방식으로, OCR의 ED·CER을 pytesseract 호출과 같은 방식으로 계산합니다. 입력은 크롭 사진 130장 `outputs/benchmarks/docunet/crop/{k}_{m} copy.png`입니다(해시: `manifest.json`). 결과는 크롭 해상도의 무손실 PNG로 저장합니다. 스캔 `outputs/dataset/files/docunet-reference-{k}.png`는 공개본 해시로 확인합니다.

LD·AD에는 SIFT flow C++(Ce Liu)가 필요하며 저장소에 포함하지 않습니다. DocUNet `eval.zip` 또는 UVDoc `eval/eval_code`의 `SIFTflow` 폴더를 직접 받아 `--siftflow-source`나 `FLATTENER_SIFTFLOW_SOURCE`로 지정하세요. 처음 실행할 때 C++ 컴파일러와 ninja로 빌드합니다. Windows에서는 `vcvars64.bat` 환경이 필요합니다.

```bash
export FLATTENER_SIFTFLOW_SOURCE=~/eval/SIFTflow
python -m flattener.benchmark docunet OUT/ --pattern "{k}_{m}.png" --output reports/docunet.json
python -m flattener.benchmark ocr OUT/ --pattern "{k}_{m}.png" --tesseract tesseract --output reports/ocr.json
python -m flattener.benchmark calibrate --reports-dir outputs/benchmarks/docunet/calibration \
  --summary outputs/benchmarks/docunet/calibration.json --paperedge-dir res_test --uvdoc-dir DocUnet_results
```

- 규약(`--convention`): `130_raw`, `130_rot64`, `128_excl64`. 기본으로 세 규약을 모두 보고합니다.
  - `130_raw`: 130장을 그대로 채점합니다. PaperEdge `adres.txt`와 공개된 DocTr·DocScanner·DocGeoNet MATLAB 스크립트가 이 방식입니다(64번 회전 코드가 주석 처리되어 있음). G1로 검증합니다.
  - `130_rot64`: 64_1·64_2를 180° 돌려 채점합니다. 위 스크립트의 회전 코드를 켠 경우이며, 어떤 게이트로도 검증하지 않았습니다.
  - `128_excl64`: 64번을 제외합니다(UVDoc). G2·G3으로 검증합니다.
  - 64번 두 장의 영향이 큽니다. PaperEdge 출력의 LD 평균은 `130_raw` 8.50, `128_excl64` 8.01입니다. 논문 수치와 비교할 때는 규약을 맞춰야 합니다.
- `--ref-resize uvdoc_view2_double`은 UVDoc 스크립트가 view 2에서 스캔을 두 번 리사이즈하는 동작을 재현합니다.
- AD는 MATLAB `lsqr(A, B)` 기본값을 따르는 LSQR(`lsqr_4iter`)로 보고합니다. 평균은 MATLAB과 맞지만(G1–G3), MATLAB을 이미지 단위로 재현하지는 않습니다. SIFT flow가 MATLAB과 같은 이미지에서도 AD가 최대 약 0.016 다릅니다(45_1). 이미지별로는 정확한 최소제곱 값(`exact_lstsq`)이 더 가까운 경우가 많습니다. 두 값을 모두 저장하고, 행마다 LSQR 반복 횟수와 종료 사유(`ad_lsqr_iterations`, `ad_lsqr_stop`)를 기록합니다.
- 지표 계산에서는 RGBA 스캔을 검정 위에, OCR에서는 흰색 위에 합성합니다. OCR 주 규약은 DocTr(Setting 1 60장, Setting 2 50장)이고 보조 규약은 UVDoc(`--protocol uvdoc`)입니다. Tesseract 버전과 실행 파일·`eng.traineddata`의 SHA-256을 기록합니다.
- OCR 수치는 Tesseract 버전에 따라 달라집니다. G4는 Tesseract 5.0.1(UB-Mannheim 5.0.1.20220118)로만 보정했고, 5.5.3에서는 ED가 3–5% 낮게 나옵니다. OCR 보고서의 `comparable`은 Tesseract 5.0.1이고 스캔 해시가 확인된 경우에만 `true`입니다. 다른 버전의 결과는 논문 수치와 비교하지 않습니다.
- `--workers` 기본값은 2이고 작업자마다 torch 스레드 1개를 씁니다. 보고서에는 규약 스위치, SIFT flow 소스 해시, 입력 파일별 해시, 환경 버전을 기록합니다.

보정 상태: 4개 게이트를 모두 통과했습니다. 원격 Windows(MSVC 빌드, 10 workers)에서 측정했고, Tesseract는 5.0.1.20220118입니다. [요약](outputs/benchmarks/docunet/calibration.json)·[보고서](outputs/benchmarks/docunet/calibration/).

| 게이트 | 대상 | 목표 | 측정 |
| --- | --- | --- | --- |
| G1 | PaperEdge 출력, `130_raw`, MATLAB `adres.txt` 평균 | 0.469963 / 8.49923 / 0.39158 | 0.469963 / 8.50032 / 0.39173 |
| G2 | PaperEdge 출력, `128_excl64` + `uvdoc_view2_double` | 0.472 / 7.98 / 0.367 | 0.47222 / 7.9819 / 0.36693 |
| G3 | UVDoc `DocUnet_results` 128장, UVDoc 규약 | 0.544 / 6.83 / 0.315 | 0.54365 / 6.8317 / 0.31488 |
| G4 | PaperEdge 출력 OCR, DocTr 규약 (ED / CER) | S1 375.60 / 0.1541, S2 777.76 / 0.2014 | S1 375.60 / 0.15410, S2 777.76 / 0.20144 |

이미지별 차이는 남습니다. G1 기준 최대 LD 0.064(32_1), AD 0.016(45_1)입니다. `examples/evaluate_docunet.py`의 값은 이 표준 수치와 비교할 수 없습니다.

## 개발과 이용 조건

빌드: `uv run --locked python -m build --no-isolation`.

코드: [Apache-2.0](LICENSE). 글꼴·의존성·데이터: [출처와 이용 조건](THIRD_PARTY.md). 배포 패키지 구성은 코드·자체 텍스트·OFL 글꼴입니다.
