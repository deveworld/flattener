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

`--quantize truncate`는 샘플링한 값을 반올림하지 않고 `floor(x*255)`로 저장합니다. UVDoc·SyntheticDoc 논문 수치는 이렇게 저장한 결과를 UVDoc 스크립트 방식(`--ref-resize uvdoc_view2_double`)으로 채점한 값이므로, 논문과 비교할 때 씁니다. 반올림과 내림만 바꿔도 LD가 0.3 정도 달라집니다. 검사, 실패 기록, 종료 코드는 기본값 `round`와 같고, 방식은 `predictions.json`과 `evidence`의 `quantize`에 남습니다. 다른 방식으로 만든 예측은 다시 쓰지 않습니다. `tools/score_docunet_local.py CKPT RUN --quantize truncate --ref-resize uvdoc_view2_double`는 이 방식으로 예측과 채점을 한 번에 하고, 설정마다 다른 파일 이름(`docunet-uvdoc_view2_double-truncate.json`)으로 저장합니다.

`--inset`은 렌더링 전에 예측한 페이지 윤곽을 안쪽으로 옮깁니다. 출력 좌표 u(−1…1)가 예측 격자의 (1−e)·u 위치를 보여 주므로, 각 변이 페이지 크기의 e/2만큼 안으로 들어옵니다. 후방 맵에 적용하므로 검사, 실패 기록, `--force-render`, 종료 코드는 그대로입니다. 값은 프리셋 이름, 두 축에 같은 값 `E`, 또는 `EX,EY`로 주고, 기본값 0은 맵을 바꾸지 않습니다. 프리셋 `syntheticdoc_half_cell`은 SyntheticDoc 추론이 121행×177열 맵을 `align_corners=True`로 키울 때 생기는 반 칸만큼의 안쪽 이동으로, x 1/177, y 1/121입니다(변마다 너비의 0.282%, 높이의 0.413%). 값은 `predictions.json`과 `evidence`의 `inset`에 남고, 다른 값으로 만든 예측은 다시 쓰지 않습니다. `uvdoc-predict`와 `tools/score_docunet_local.py`도 `--inset`을 받고, 도구는 `-inset-syntheticdoc_half_cell`을 붙인 파일 이름으로 저장합니다.

`--rerun R`은 사진 전체로 한 번 예측한 뒤 R번 더 잘라서 다시 예측합니다. 매번 현재 맵의 페이지 윤곽(격자 가장자리 점)을 원본 픽셀로 옮겨 축에 맞춘 상자를 잡고, 변마다 페이지 크기의 4%(학습 크롭 여백 0–8%의 가운데)를 더해 정수 픽셀로 넓힌 뒤 이미지 안으로 자릅니다. 그 크롭을 모델 입력 크기로 줄여 다시 예측하고, 새 격자를 크롭 위치와 합성해 원본 좌표로 되돌리므로 최종 렌더링은 원본을 한 번만 샘플링합니다. `--deskew`(`--rerun`과 함께)는 윤곽의 위아래 변 방향으로 페이지 각도를 추정하고(선분 방향의 원형 평균에 가운데 절반의 평균 편차를 더함), 추정이 쓸 만하고 45° 이하이면 크롭 전체가 들어가는 캔버스에서 페이지를 똑바로 돌린 뒤 예측합니다. 캔버스에서 크롭 바깥은 크롭 가장자리 픽셀을 반복합니다. 변이 한 점으로 줄었거나(`degenerate_outline`), 변 방향이 서로 맞지 않거나(`inconsistent_edges`), 45°를 넘으면(`angle_above_limit`) 돌리지 않습니다. 기본값 0은 지금 동작 그대로입니다. 최종 맵의 검사, 실패 기록, `--force-render`, 종료 코드는 같고, 설정은 `predictions.json`과 `evidence`의 `rerun`에, 이미지마다의 상자, 각도, 건너뛴 이유는 `crops`에 남습니다. 다른 설정으로 만든 예측은 다시 쓰지 않습니다. `dir300-predict`, `uvdoc-predict`, `tools/score_docunet_local.py`도 같은 옵션을 받고, 도구는 `-rerun1`, `-rerun1-deskew`를 붙인 파일 이름으로 저장합니다(`--stress`와는 함께 쓰지 않습니다).

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
- OCR 보고서의 `comparable`은 G4를 통과한 Tesseract 빌드로 읽고 스캔 해시가 맞을 때만 `true`입니다. 빌드는 버전 문자열, Leptonica 버전, 실행 파일 SHA-256, `eng.traineddata` SHA-256으로 구분하며 목록은 `flattener.benchmark.ocr.CALIBRATED_TESSERACT_BUILDS`에 있습니다. 지금은 UB-Mannheim Windows 빌드 `v5.0.1.20220118`(Leptonica 1.78.0, tessdata_fast `eng`) 하나입니다.
- Linux에서는 이 설치 파일(`tesseract-ocr-w64-setup-v5.0.1.20220118.exe`)을 `7z x`로 풀고 Wine으로 실행합니다: `--tesseract TREE/tesseract.exe --launcher wine`. `WINEPREFIX`와 `WINEDEBUG=-all`은 환경 변수로 줍니다. Wine 11.0(AVX2 CPU)에서 G4의 OCR 문자열 114개가 Windows(AVX-512 CPU) 실행과 모두 같았습니다.
- conda-forge `tesseract=5.0.1`(Leptonica 1.82.0)은 G4를 통과하지 못해 비교 가능한 수치를 내지 않습니다(S1 ED 384.47 / CER 0.1593, S2 809.06 / 0.2079, 목표보다 2.4–4.0% 높음). 이 빌드는 `--tessdata-dir`로 `TESSDATA_PREFIX`를 지정해야 돌아갑니다.
- G1–G4 보정 결과는 `outputs/benchmarks/docunet/calibration.json`에 있습니다.

실사진 스트레스 평가는 DocUNet 사진을 바꿔 가며 같은 평가기로 채점합니다.

```bash
python -m flattener.benchmark docunet-stress --weights CKPT --output-dir OUT --force-render --device cuda
```

- 변형은 `crop`(표준 크롭, 같은 실행 안의 기준), `full_frame`(배경까지 담긴 원본 사진 `docunet-photo-{k}_{m}.jpg`, EXIF 방향 적용), `rot90`·`rot270`(크롭을 ±90° 돌림), `small_page`(무늬 캔버스 면적의 10–25%를 차지하는 크롭)입니다. 출력은 모두 똑바로 선 크롭의 크기로 렌더링합니다.
- 스캔은 똑바로 둔 채 채점하므로 돌아간 페이지를 그대로 돌려주는 모델은 `rot90`·`rot270` 점수가 낮게 나옵니다.
- 결과는 `OUT/stress.json`과 변형별 `OUT/<변형>/docunet.json`에 남습니다. 검사에 실패한 예측 수와 기록에 없는 예측 수도 변형별로 셉니다.
- 다시 실행하면 가중치와 입력 픽셀이 같은 예측만 남기고 나머지는 새로 만듭니다. 이미지별 점수는 예측 해시, 규약, SIFT flow 빌드, 스캔 해시가 모두 같을 때만 재사용합니다.
- UVDoc 공개 가중치의 기준선은 `outputs/benchmarks/docunet/stress/uvdoc-mit/`에 있습니다(`external_pretrained`). 130_raw MS-SSIM/LD는 `crop` 0.540/7.10, `full_frame` 0.205/54.5, `rot90` 0.307/32.3, `rot270` 0.307/30.1, `small_page` 0.310/43.4이고, 검사에 실패한 예측은 없었습니다. `full_frame`은 EXIF 방향을 적용한 뒤 CPU에서 다시 채점한 값입니다. 처음 값 0.204/51.9는 사진 30장(방향 태그 6)을 옆으로 누운 채 넣은 결과입니다.

UVDoc 벤치마크(실사진 50장)는 UVDoc 공개 코드(`uvdocBenchmark_pred.py`, `evalScriptUVDoc.m`, MIT)의 규약으로 예측하고 채점합니다.

```bash
python -m flattener.benchmark uvdoc-prepare UVDoc_benchmark.zip
python -m flattener.benchmark uvdoc-predict --weights CKPT --output-dir OUT
python -m flattener.benchmark uvdoc OUT --output reports/uvdoc.json
```

- 자료는 https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip (775 MB)에서 받습니다. `uvdoc-prepare`는 압축 파일의 SHA-256을 확인하고 `outputs/benchmarks/uvdoc-bench/`에 푼 뒤 `manifest.json`(출처 URL, 압축 파일과 파일별 해시)을 씁니다.
- 예측: 모델은 사진 `img/{id}.png`을 보고, 예측한 격자는 `warped_textures/{id}.png`에 그 크기로 적용합니다. 이 텍스처는 평평한 텍스처를 사진과 똑같이 휘어 놓은 것으로, 조명과 배경이 없습니다. 쌍선형 보간, 원본 바깥은 0, `floor(x*255)` 저장까지 공개 코드와 같습니다. 결과는 `OUT/uwp_texture/`, 같은 격자를 사진에 적용한 결과는 `OUT/uwp_img/`에 둡니다. 모든 이미지를 저장하고 `flattener.render` 검사 결과는 `OUT/predictions.json`에 남깁니다.
- 채점: `uwp_texture`(기본값, `--pred-type`)를 평평한 텍스처 `texture_sample/{id}.png`와 비교합니다. DocUNet과 같은 MATLAB 에뮬레이션(회색 변환, 텍스처를 598,400화소로 한 번 리사이즈, MS-SSIM, SIFT flow, AD, LD)을 쓰고, 50장 전체의 평균을 냅니다. 비교 기준은 UVDoc README erratum의 50장 값(UVDoc 0.785 / AD 0.119)입니다. 논문 본문 표는 절반만 쓴 값입니다.
- 평가기 검증: UVDoc이 공개한 결과 이미지(`UVDocBenchmark_results.zip`)는 MS-SSIM 0.7854, AD 0.1186, LD 6.812로 erratum 값과 AADD 표 S1의 LD 6.810에 맞습니다. UVDoc MIT 가중치를 CPU에서 다시 예측하면 공개 이미지와 화소값이 최대 1 다르고, 점수는 0.7854 / 0.1201 / 6.890입니다. LD·AD 차이는 대부분 샘플 00040 한 장의 SIFT flow에서 나옵니다(LD +2.87, AD +0.079).

DIR300(실사진 300장, OCR 90장)은 DocGeoNet의 공개 규약(`ssim_ld_eval_DIR300.m`, `OCR_eval_DIR300.py`)으로 채점합니다. DocGeoNet 코드는 비상업 조건이라 규약 설명을 보고 직접 구현했습니다.

```bash
python -m flattener.benchmark dir300-predict --weights CKPT --output-dir OUT --force-render
python -m flattener.benchmark dir300 OUT --output reports/dir300.json
python -m flattener.benchmark dir300-ocr OUT --output reports/dir300-ocr.json
```

- 자료는 `outputs/benchmarks/dir300/`의 `dist300/`(입력 사진)과 `gt300/`(평평한 기준 이미지)이고, 비상업 평가 전용입니다(`LICENSE.md`, `provenance.json`). 두 폴더는 공식 파일과 해시가 같아야 합니다(`--no-verify-inputs`, `--no-verify-references`로 끕니다). 공식 파일은 248번 기준 이미지를 `248.PNG`로 저장해 두어서 대소문자를 가리지 않고 찾습니다.
- 예측: 사진마다 원래 크기로 보정해 `OUT/{k}.png`에 씁니다. 가장 큰 사진(3472×4624)이 렌더러 기본 한도 1,600만 화소를 넘기 때문에 DIR300 예측만 한도를 2,000만 화소로 둡니다.
- 채점: 보정 결과 `k`를 `gt300/{k}.png`와 짝짓습니다. 둘 다 회색으로 바꾼 뒤 기준 이미지를 `sqrt(598400/H/W)`배로(DocUNet과 같은 598,400화소), 보정 결과를 그 크기로 리사이즈합니다. MS-SSIM·LD는 DocUNet `evalUnwarp.m`, AD는 PaperEdge `evalAlignedUnwarp.m`과 같고, 300쌍의 단순 평균입니다. DocGeoNet 공개 결과처럼 `{k}_rec.png`로 저장한 폴더는 `--pattern "{k}_rec.png"`로 채점합니다.
- OCR: DocGeoNet 목록(`ocr_img_DIR300.txt`)의 90장을 pytesseract 기본값으로 읽고, ED와 CER(= ED / 기준 문자열 길이)의 단순 평균을 냅니다. 보정 결과는 저장된 크기 그대로 읽습니다. `comparable` 조건은 DocUNet OCR과 같습니다.
- DIR300 사진에는 배경이 함께 찍혀 있습니다. 사진 전체를 한 번에 펴는 단일 단계 모델은 페이지를 먼저 찾거나 잘라 다시 돌리는 방법보다 점수가 크게 낮고, 논문 표에는 두 방식이 섞여 있습니다. 예를 들어 UVDoc은 전처리 없이 0.622 / 7.73 / 0.218(TADoc 표), DocRes 위치 찾기를 붙이면 0.676 / 4.83 / 0.165(Q-DRFormer)입니다.
- `tools/score_dir300_local.py CKPT RUN [--quantize truncate] [--ocr]`는 예측과 채점을 한 번에 하고 `outputs/dewarp-runs/RUN/dir300*.json`에 저장합니다. 예측 이미지(장당 약 9 MB)는 `~/.cache/flattener/dir300/`에 둡니다.
- 평가기 검증: UVDoc MIT 가중치로 사진마다 원래 크기로 예측하고 `floor(x*255)`로 저장하면(`--quantize truncate`) 0.6215 / 7.742 / 0.2183으로, TADoc 표의 0.622 / 7.73 / 0.218과 MS-SSIM −0.0005, LD +0.012, AD +0.0003 차이입니다. 보고서는 `outputs/dewarp-runs/uvdoc-mit/dir300-truncate.json`에 있습니다(`external_pretrained`).

## 결과의 근거 구분

`predictions.json`, 학습 로그·`status.json`·`best.json`, `flattener.benchmark` 보고서에는 `evidence` 항목이 들어갑니다. `evidence_kind` 하나로 결과의 종류를 구분합니다.

- `new_model_real_photo`: 직접 학습한 체크포인트(1만 스텝 이상)로 DocUNet 실사진(표준 크롭 또는 스트레스 변형), UVDoc 벤치마크 사진, DIR300 사진을 보정하고 채점한 결과입니다. 새 모델의 실사진 성능 향상으로 세는 결과는 이것 하나입니다.
- `gt_crop_validation`, `gt_full_frame_validation`: 학습 중 검증입니다. 정답이 있는 샘플을 정답 페이지 영역에 맞춰 자르거나 전체 화면으로 넣어 잽니다. 학습 상태를 확인하는 용도입니다.
- `external_pretrained`: UVDoc 공개 가중치, 논문 저자가 공개한 결과처럼 밖에서 학습한 모델의 결과입니다. 위 예제의 UVDoc 가중치와 기준선이 여기에 들어가며, 비교 기준으로 씁니다.
- `smoke_training`: 1만 스텝 미만으로 학습한 체크포인트의 결과입니다. 코드가 도는지 확인하는 용도입니다.
- `docunet_standard`, `docunet_stress`, `uvdoc_benchmark`, `dir300_standard`: 예측 폴더에 `predictions.json`이 없어 모델 출처를 모르거나, 예전 좌표 모델(`own_previous_model`)을 채점한 결과입니다.
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
