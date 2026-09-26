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

`--print-config`는 최종 설정만 출력하고 끝냅니다. 학습은 출력 폴더의 마지막 체크포인트에서 이어 가고(`--resume auto`), 체크포인트는 `RUN/checkpoints/step_*.pt`에 쌓입니다. 가장 좋은 검증 결과는 `best.json`이 가리킵니다. `--weights`에는 UVDoc 공개 가중치 `best_model.pkl`도 넣을 수 있습니다.

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
