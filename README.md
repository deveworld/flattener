# flattener

문서 사진의 휘어짐과 원근을 펴는 Python 라이브러리입니다. 실험 단계이며, 추론 결과 메타데이터의 `quality_status`는 `"not_evaluated"`입니다.

## 설치

Linux x86_64, Python 3.12 기준입니다. `uv.lock`은 CPU용 PyTorch를 고정합니다.

```bash
uv sync --locked --python 3.12
uv run --locked pytest -q            # 학습·역전파 테스트까지: --run-training
uv run --locked ruff check .
uv run --locked python -m build --no-isolation
```

## 렌더링

```python
from flattener import identity_map, load_image, render, save_result

image = load_image("page.jpg")
result = render(image, identity_map(image.shape[:2]))
save_result(result, "out/page")
```

`render`는 출력 픽셀마다 원본의 어느 위치를 읽을지 담은 좌표 맵을 받습니다. 맵은 `float32 (H, W, 2)`이고 마지막 축은 `(x, y)`, 값은 원본 기준 정규화 좌표입니다. 크기는 언제나 `(height, width)` 순서입니다. 좌표 공식, 마스크, 실패 처리는 [인터페이스 사양](docs/INTERFACE_SPEC.md)에 있습니다.

## 현재 기본 모델

```python
from flattener import RenderConfig
from flattener.learning.coordinate_study import load_coordinate_predictor
from flattener.learning.inference import unwarp

predictor = load_coordinate_predictor("outputs/coordinate-fold/selected.pt")
result = unwarp(image, predictor, (1024, 768), config=RenderConfig(on_error="return"))
```

- `selected.pt`는 global 구조에 입력 192×144, 좌표 격자 9×7인 모델입니다(SHA-256 `ee7e2cf7…`).
- 새 모델이 표준 벤치마크에서 이 모델을 앞설 때까지 기본 모델로 둡니다.
- ±90° 돌아간 페이지에서는 방향을 잘못 잡고 페이지 일부만 확대하는 일이 잦습니다.
- `success`는 기하 검사와 렌더링을 통과했다는 뜻입니다. 글자 잘림과 남은 휨은 결과 이미지에서 확인합니다.
- 사진 한 장을 보정하는 스크립트: `python examples/unwarp_photo.py --help`.

## 새 모델 (flattener.dewarp)

UVDoc 계열의 격자 회귀 모델입니다. 학습 설정은 JSON 파일 하나에 담고, 명령줄 옵션으로 일부 값을 덮어씁니다. `examples/dewarp/`의 설정에 적힌 데이터·출력 경로는 자기 환경에 맞게 고쳐서 씁니다.

```bash
python -m flattener.dewarp.train --config examples/dewarp/train-release-uvdoc-init.json
python -m flattener.dewarp.predict --weights CKPT image --input a.jpg --output a.png
python -m flattener.dewarp.predict --weights CKPT docunet --input-dir crop --output-dir OUT
```

예측은 `flattener.render`의 검사(좌표 범위, 덮는 비율, 접힘)를 거칩니다. 검사에 실패한 사진은 저장하지 않고 상태(`invalid_transform`과 이유)를 출력과 `predictions.json`에 남기며, 명령은 종료 코드 1로 끝납니다. `--force-render`를 붙이면 실패한 사진도 저장합니다. 공개 방법들과 같은 방식으로 벤치마크를 채점할 때 씁니다. 이때도 실패 기록과 종료 코드 1은 그대로이고, `flattener.benchmark` 보고서의 `prediction_failures`가 실패한 사진 수를 셉니다.

검증은 화면 종류별로 EPE를 따로 기록합니다. 기본 화면은 페이지에 꼭 맞춘 크롭(`page`), DocUNet처럼 여백을 둔 크롭(`docunet`), 사진 전체(`full_frame`), 크롭을 ±90° 돌린 화면(`rot90`, `rot270`), 무늬 있는 캔버스의 10–25%만 차지하는 작은 페이지(`small_page`)입니다. 최고 체크포인트는 `val_select`로 정한 화면 하나로 고르며, 정하지 않으면 첫 화면(`page`)을 씁니다. 정답 격자에 유효한 점이 하나도 없는 샘플은 평균에서 빼고 `not_evaluable`로 셉니다.

`--print-config`는 최종 설정만 출력하고 끝냅니다. 학습은 출력 폴더의 마지막 체크포인트에서 이어 가고(`--resume auto`), 체크포인트는 `RUN/checkpoints/step_*.pt`에 쌓입니다. 가장 좋은 검증 결과는 `best.json`이 가리킵니다. 검증 설정이 바뀐 채로 이어 가면 이전 `best.json`은 `best_before_step_*.json`으로 옮겨지고 새 기준으로 다시 고릅니다. `--weights`에는 UVDoc 공개 가중치 `best_model.pkl`도 넣을 수 있습니다.

- `release` 트랙은 이용 조건이 확인된 데이터만 받습니다. 배포 대상은 이 트랙의 가중치입니다.
- `research` 트랙은 Doc3D·UVDoc 데이터셋처럼 비상업 조건이 붙은 자료로 학습하며, 가중치는 로컬 비교에만 씁니다.

## 표준 벤치마크 평가

`flattener.benchmark`는 DocUNet의 MS-SSIM·LD·AD를 공개 MATLAB 스크립트와 같은 방식으로, OCR의 ED·CER을 pytesseract 호출과 같은 방식으로 계산합니다. 기본 입력은 크롭 사진 `outputs/benchmarks/docunet/crop/`과 스캔 `outputs/dataset/files/docunet-reference-{k}.png`입니다.

```bash
export FLATTENER_SIFTFLOW_SOURCE=~/eval/SIFTflow
python -m flattener.benchmark docunet OUT/ --pattern "{k}_{m}.png" --output reports/docunet.json
python -m flattener.benchmark ocr OUT/ --pattern "{k}_{m}.png" --output reports/ocr.json
python -m flattener.benchmark calibrate --reports-dir reports/calibration \
  --paperedge-dir res_test --uvdoc-dir DocUnet_results
```

- LD·AD에는 SIFT flow C++ 소스(Ce Liu)가 필요합니다. DocUNet `eval.zip`이나 UVDoc `eval_code`의 `SIFTflow` 폴더를 받아 `--siftflow-source` 또는 `FLATTENER_SIFTFLOW_SOURCE`로 지정합니다. 처음 실행할 때 C++ 컴파일러와 ninja로 빌드합니다.
- `--convention`은 `130_raw`(PaperEdge, DocTr 계열), `128_excl64`(UVDoc), `130_rot64` 중에서 고르고, 기본값은 셋 모두입니다. 논문 수치와 비교할 때는 규약을 맞춥니다. `--ref-resize uvdoc_view2_double`은 UVDoc 스크립트가 view 2에서 스캔을 두 번 리사이즈하는 동작을 재현합니다.
- AD는 MATLAB `lsqr` 기본값을 따르는 `lsqr_4iter`로 보고하고, 정확한 최소제곱 값 `exact_lstsq`도 함께 저장합니다.
- OCR 보고서의 `comparable`은 Tesseract 5.0.1이고 스캔 해시가 맞을 때만 `true`입니다.
- G1–G4 보정 결과는 `outputs/benchmarks/docunet/calibration.json`에 있습니다.

실사진 스트레스 평가는 DocUNet 사진을 바꿔 가며 같은 평가기로 채점합니다.

```bash
python -m flattener.benchmark docunet-stress --weights CKPT --output-dir OUT --force-render --device cuda
```

- 변형은 `crop`(표준 크롭, 같은 실행 안의 기준), `full_frame`(배경까지 담긴 원본 사진 `docunet-photo-{k}_{m}.jpg`, EXIF 방향 적용), `rot90`·`rot270`(크롭을 ±90° 돌림), `small_page`(무늬 캔버스 면적의 10–25%를 차지하는 크롭)입니다. 출력은 모두 똑바로 선 크롭의 크기로 렌더링합니다.
- 스캔은 똑바로 둔 채 채점하므로 돌아간 페이지를 그대로 돌려주는 모델은 `rot90`·`rot270` 점수가 낮게 나옵니다.
- 결과는 `OUT/stress.json`과 변형별 `OUT/<변형>/docunet.json`에 남습니다. 검사에 실패한 예측 수와 기록에 없는 예측 수도 변형별로 셉니다.
- 다시 실행하면 가중치와 입력 픽셀이 같은 예측만 남기고 나머지는 새로 만듭니다. 이미지별 점수는 예측 해시, 규약, SIFT flow 빌드, 스캔 해시가 모두 같을 때만 재사용합니다.
- UVDoc 공개 가중치의 기준선은 `outputs/benchmarks/docunet/stress/uvdoc-mit/`에 있습니다(`external_pretrained`). 130_raw MS-SSIM/LD는 `crop` 0.540/7.10, `rot90` 0.307/32.3, `rot270` 0.307/30.1, `small_page` 0.310/43.4이고, 검사에 실패한 예측은 없었습니다. `full_frame`의 0.204/51.9는 EXIF 방향을 적용하기 전 값입니다. 사진 30장(방향 태그 6)이 옆으로 누운 채 들어가 있어 다시 채점해야 합니다.

## 결과의 근거 구분

`predictions.json`, 학습 로그·`status.json`·`best.json`, `flattener.benchmark` 보고서에는 `evidence` 항목이 들어갑니다. `evidence_kind` 하나로 결과의 종류를 구분합니다.

- `new_model_real_photo`: 직접 학습한 체크포인트(1만 스텝 이상)로 DocUNet 실사진(표준 크롭 또는 스트레스 변형)을 보정하고 채점한 결과입니다. 새 모델의 실사진 성능 향상으로 세는 결과는 이것 하나입니다.
- `gt_crop_validation`, `gt_full_frame_validation`: 학습 중 검증입니다. 정답이 있는 샘플을 정답 페이지 영역에 맞춰 자르거나 전체 화면으로 넣어 잽니다. 학습 상태를 확인하는 용도입니다.
- `external_pretrained`: UVDoc 공개 가중치, 논문 저자가 공개한 결과처럼 밖에서 학습한 모델의 결과입니다. 위 예제의 UVDoc 가중치와 기준선이 여기에 들어가며, 비교 기준으로 씁니다.
- `smoke_training`: 1만 스텝 미만으로 학습한 체크포인트의 결과입니다. 코드가 도는지 확인하는 용도입니다.
- `docunet_standard`, `docunet_stress`: 예측 폴더에 `predictions.json`이 없어 모델 출처를 모르거나, 예전 좌표 모델(`own_previous_model`)을 채점한 결과입니다.
- `evaluator_calibration`은 평가기 보정(G1–G4) 보고서, `predictions`는 점수 없이 예측만 담은 기록, `training_loss`는 학습 로그와 `status.json`의 학습 손실입니다. `calibration.json`은 목록에 올린 보고서마다 `evidence`를 붙이고, 그전에 만든 보고서 파일은 해시가 기록돼 있어 그대로 둡니다.

`measurement`(무엇을 쟀는지), `model_provenance`(`own_checkpoint`, `own_previous_model`, `external_pretrained`), `checkpoint_step`, `initialization`도 함께 기록합니다. `initialization`은 직접 학습한 체크포인트의 출발점(`scratch` 또는 UVDoc 가중치인 `external_pretrained`)입니다. UVDoc 가중치에서 시작한 미세조정도 `new_model_real_photo`로 세고, 향상 폭은 같은 변형의 UVDoc 기준선과 비교해 읽습니다.

## 데이터와 내부 평가

학습·평가 자료는 `outputs/dataset`에 둡니다. `manifest.json`에 경로·분할·출처·해시를 기록하고, 원본 파일은 `files/`에 있습니다. SmartDoc 촬영 템플릿 30종과 사진 360장으로 만든 휨 합성 자료 2,400장은 `--include-paper-augmentation`을 붙여야 포함됩니다. 촬영 사진의 정답은 네 꼭짓점으로 만든 평면 근사입니다. `examples/generate_paired_scenes.py OUTPUT`은 작은 합성 세트를 새로 만들며, 기본값이 달라서 `outputs/dataset`과는 별개의 세트가 나옵니다.

```bash
python -m flattener.learning.evaluation --dataset outputs/dataset \
  --checkpoint outputs/coordinate-fold/selected.pt --split test \
  --output outputs/evaluation --output-size 192 144 --include-paper-augmentation
```

검증 세트는 `--split validation`으로 평가합니다. 후보 체크포인트 선택은 `python -m flattener.learning.evaluation select --help`를 봅니다.

## 좌표 모델 학습

옵션은 `python -m flattener.learning.coordinate_study {check-device,inspect,train} --help`로 확인합니다.
- 장치는 `--device cpu|cuda|xpu`로 지정한 것만 씁니다. XPU는 `torch.xpu.is_available()`이 참인 XPU용 PyTorch 환경에서 돌립니다.
- `--resume OUTPUT/last.pt`는 같은 코드·환경·플래그일 때 이어 갑니다. 데이터·코드·환경이 바뀌었으면 `--rebuild-cache`로 전처리 캐시를 다시 만듭니다.

## 라이선스

코드는 [Apache-2.0](LICENSE)입니다. UVDoc에서 가져온 `flattener.dewarp.model`은 MIT, 함께 배포하는 글꼴은 OFL-1.1입니다. 외부 코드와 데이터의 출처·이용 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다. `benchmarks/benchmark_renderer.py`는 렌더러 속도 스모크 테스트입니다.
