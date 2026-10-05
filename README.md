# flattener

문서 사진 한 장에서 페이지를 찾아 똑바로 세우고, 휨을 편 뒤 원본을 한 번만 샘플링해 렌더링하는 스캐너입니다.
코드는 Apache-2.0입니다. 가중치를 공개하면 받는 곳을 여기에 적습니다.
결과 기록의 `quality_status`는 `"not_evaluated"`입니다. 결과 이미지는 직접 확인합니다.

## 설치

Linux x86_64, Python 3.12, CPU용 PyTorch 기준입니다. 버전과 해시는 `uv.lock`에 고정돼 있습니다.

```bash
uv sync --locked --python 3.12
uv run --locked pytest -q            # 학습 테스트까지: --run-training
uv run --locked ruff check .
uv run --locked python -m build --no-isolation
```

## 사진 스캔

```bash
python -m flattener.scan IMG... --out DIR --bundle bundle.json
python -m flattener.scan IMG... --out DIR --weights DEWARP.pt [--gate GATE.pt] [--orient-weights O.pt]
```

- 가중치는 두 방식 중 하나로만 줍니다. 번들은 `{"schema": "flattener-scan-bundle-1", "dewarp": ..., "gate": ..., "orient": ...}`이고 `dewarp`만 꼭 있어야 합니다. 각 값은 번들 파일 기준 경로이거나 `{"path": ..., "sha256": ...}`입니다. `dewarp`에는 `"tta": "flip"`(추론 증강, 아래 "모델")을, `orient`에는 실사진으로 정한 방향 판별 임계값 `"threshold"`를 넣을 수 있고, `--orient-threshold`를 주면 그 값을 씁니다. 그 밖의 키는 오류입니다.
- 가중치는 배포 검사(아래 "모델")를 통과해야 불러옵니다. `--research`를 주면 통과하지 못한 가중치도 불러오고, 기록에 `research: true`가 남습니다.
- `--device`의 기본값은 `cpu`입니다.
- 수동 수정: `--region L,T,R,B`, `--quad x1,y1,...,x4,y4`(왼쪽 위부터 시계 방향), `--quarter 0..3`, `--no-dewarp`, `--long-side N`, `--output-size H,W`, `--aspect`.
- 편 이미지에는 기본으로 샤프닝을 겁니다(`--sharpen`, 기본 1.0, `--sharpen 0`이면 끔). 밝기만 선명하게 하므로 색은 그대로이고, 실패해서 돌려주는 원본에는 걸지 않습니다. 기본 모델이 편 벤치마크 결과에 걸어 보면 DocUNet 글자 인식 오류율(CER, DocTr 60장)이 0.158에서 0.119로 줄었습니다. DIR300은 편집 거리 평균이 406에서 361로 줄었고, CER 평균은 정답 스캔에서 175자만 읽히는 사진 한 장 때문에 0.141에서 0.184로 올랐습니다(그 사진을 빼면 0.118에서 0.103).
- 모든 옵션: `python -m flattener.scan --help`.

```python
from flattener.scan.io import save_scan
from flattener.scan.pipeline import Scanner

result = Scanner.load("bundle.json").scan("photo.jpg")
save_scan(result, "out", "photo")  # out/photo.jpg, out/photo.scan.json
```

`result.outcome`은 아래 결과 종류 중 하나이고, `result.image`는 RGB `uint8` 배열입니다.

## 결과와 실패

- 사진마다 `<stem>.jpg`(JPEG 품질 95, 4:4:4)와 기록 `<stem>.scan.json`을 씁니다. 펼친 책의 두 쪽이 찍힌 사진은 접힌 곳에서 나눠 `<stem>-p1`, `<stem>-p2`로 따로 씁니다. 번호는 왼쪽 쪽부터이고, `--page-order rtl`이면 오른쪽 쪽부터입니다. `--pages 1`은 나누지 않고, `--pages 2`는 항상 나눕니다. PNG는 `--format png`입니다. `--debug`는 페이지 윤곽을 그린 사진을, `--timing F.jsonl`은 사진별 단계 시간을 한 줄씩 씁니다.
- `outcome`은 `rectified`, `rectified_with_warnings`, `perspective`, `original_returned`, `failed` 중 하나입니다. 사유 코드와 한/영 안내가 기록에 남습니다.
- 모든 후보가 실패하면 `--fallback original`(기본값)은 원본을 돌려주고, `--fallback perspective`는 페이지 꼭짓점이 확실할 때 원근만 보정합니다. `--on-error original`(기본값)은 실패한 사진의 원본을 저장하고, `--on-error return`은 이미지를 저장하지 않습니다.
- 방향 판별 가중치가 없으면 EXIF를 적용한 사진 방향을 그대로 쓰고 `orientation_uncertain` 경고를 남깁니다.
- 종료 코드: 모두 스캔되면 0, 대체 결과나 실패가 하나라도 있으면 1, 사용 오류는 2.

## 모델

- 휨 보정: `~/.cache/flattener/scan-weights/soup-c70-ck/step_00040000.pt`와 옆의 `.pt.json`입니다. UVDoc 구조에 파라미터 약 800만 개입니다. 공개해도 되는 합성 페이지로 처음부터 학습한 모델에 기울어진 카메라·배경·옆 쪽 합성을 더해 `dewarp-v2`를 만들었습니다. 그다음 UVDoc의 원본 촬영분(구겨지고 접힌 빈 종이 사진, MIT)에 공개해도 되는 페이지를 입힌 자료를 절반 섞어 4만 스텝 학습했습니다. 여기서 두 갈래로 4만 스텝씩 더 학습해 반반 섞었습니다(`tools/soup.py`). 한쪽은 렌더 명암을 걷어낸 페이지와 무늬 배경 자료를 70% 섞었고(`train-release-uvdoc-clean70-ft.json`), 다른 쪽은 같은 자료 50%에 주름·접힘 렌더 15%를 더했습니다(`train-release-uvdoc-clean-crumple-ft.json`). 주름 렌더 쪽만 쓰면 책 사진 한 장에서 맞은편 페이지가 따라 들어오는데, 섞으면 사라집니다. `dewarp-v2`보다 DocUNet(0.573 / 5.42 / 0.240 → 0.581 / 5.16 / 0.228), UVDoc 벤치(0.652 / 7.11 / 0.211 → 0.739 / 4.69 / 0.103), DIR300(0.643 / 6.70 / 0.255 → 0.699 / 4.31 / 0.159)이 모두 좋습니다. KR-Real에서는 2장이 나빠졌습니다. 펼친 책 한 장이 두 쪽으로 나뉘지 않고, 페이지가 사진 밖까지 이어지는 가까운 사진 한 장은 원본을 돌려줍니다.
- 추론 증강(TTA): 휨 보정 모델을 부를 때마다 사진과 좌우를 뒤집은 사진을 한 묶음으로 예측하고, 뒤집은 쪽 격자를 되돌려 평균합니다. 두 격자가 이미지 변의 3%보다 크게 갈리면(맞은편 페이지나 잘린 페이지처럼 페이지 범위가 애매할 때) 평균하지 않고 원래 사진의 격자만 씁니다. 꼼꼼하게 펴는 설정(`--rerun 2 --rect`)에서 DocUNet 0.581 / 4.98 / 0.214 → 0.589 / 4.91 / 0.208, DIR300 0.714 / 3.95 / 0.133 → 0.718 / 3.78 / 0.126, UVDoc 벤치 0.753 / 4.80 / 0.109 → 0.767 / 4.53 / 0.088입니다. KR-Real 결과 종류는 그대로이고, 노트북 CPU 2스레드에서 장당 시간 중앙값이 1.43초에서 1.86초로 늘어납니다. 예측 명령에서는 `--tta flip`입니다.
- 세 모델을 함께 쓰는 번들: `~/.cache/flattener/scan-weights/bundle-v9.json`(휨 보정은 위 모델에 `"tta": "flip"`). TTA 없이 빠르게 쓰려면 같은 모델의 `bundle-v8.json`을 씁니다. 가중치와 계보의 조상 체크포인트는 비공개 Hugging Face 저장소 `DevWorld/flattener`에 백업되어 있습니다. 다시 만드는 순서는 [휨 보정 학습](examples/dewarp/README.md)에 있습니다.
- 페이지 찾기(페이지 게이트 v1.2): `~/.cache/flattener/scan-weights/page-gate-v1.2/best.pt`와 옆의 `.sha256`입니다. 파라미터 약 56만 개이고, 입력은 192×144입니다. 공개해도 되는 합성 장면(SyntheticDoc 페이지, 직접 만든 한국어 페이지, 노트북·키보드·화면·흰 가구·밑에 깔린 문서가 있는 책상, 가까이 찍은 페이지)과 SmartDoc 촬영 영상 프레임(CC BY 4.0)으로 처음부터 학습했습니다(`examples/gate/train-gate-release-v1.2.json`). SmartDoc 테스트 문서 사진 72장의 IoU는 0.977(v0 0.939)입니다. KR-Real에서는 v0보다 4장이 좋아지고 2장이 나빠졌습니다.
- 이전 페이지 찾기(페이지 게이트 v0): `python tools/locator_v0.py install`이 `$FLATTENER_CACHE_DIR/scan-weights`(기본 `~/.cache/flattener/scan-weights`)에 `page-gate-v0.pt`를 씁니다. `python tools/locator_v0.py reproduce`는 SmartDoc 사진 48장에서 기록된 IoU 0.951을 다시 잽니다.
- 배포 검사: 기록된 트랙이 모두 release이고, 계보가 처음부터(scratch) 시작하고, 모든 학습 단계의 데이터 spec이 release 조건을 통과해야 배포할 수 있습니다. `python -m flattener.scan.shippable CKPT...`는 모두 통과하면 0, 하나라도 아니면 1, 사용 오류면 2로 끝납니다. 휨 보정 가중치에 씁니다. 페이지 게이트는 스캐너가 불러올 때 직접 판정합니다.
- 방향 판별(OrientNet-1 v3): `~/.cache/flattener/scan-weights/orient-v3/step_00004000.pt`와 옆의 `.pt.json`입니다. 파라미터 약 44만 개입니다. 공개해도 되는 합성 페이지로 처음부터 학습한 v1을 브라우저에서 인쇄한 문서(쪽수가 위에 있는 문서)와 두 쪽 펼침을 더한 자료로 4천 스텝 더 학습했습니다(`examples/orient/train-orient-release-v3.json`). 합성 검증 정확도는 99.5%입니다. KR-Real에서 방향을 판단한 사진 70장 중 69장이 맞았습니다(v1은 71장 중 67장). 임계값은 0.9입니다.
- 학습 방법: [휨 보정](examples/dewarp/README.md), [방향 판별](examples/orient/README.md), [페이지 찾기](examples/gate/README.md).

## 렌더러

`render(image, backward_map)`는 출력 픽셀마다 원본의 어느 위치를 읽을지 담은 좌표 맵(`float32 (H, W, 2)`, 마지막 축 `(x, y)`, 원본 기준 정규화 좌표)으로 원본을 한 번 샘플링합니다.
좌표 공식, 마스크, 실패 처리는 [인터페이스 사양](docs/INTERFACE_SPEC.md)에 있습니다.

## 평가

```bash
python -m flattener.benchmark --help   # docunet, ocr, calibrate, docunet-stress, uvdoc-*, dir300*
python -m flattener.benchmark docunet PRED_DIR --pattern "{k}_{m}.png" --output docunet.json
```

- 입력은 `outputs/benchmarks/docunet/`(`crop/`, `reference/`, `photo/`), `outputs/benchmarks/uvdoc-bench/`, `outputs/benchmarks/dir300/`에 둡니다.
- LD와 AD에는 SIFT flow 소스가 필요합니다. `FLATTENER_SIFTFLOW_SOURCE` 또는 `--siftflow-source`로 줍니다.
- OCR 점수는 Tesseract 5.0.1.20220118(UB-Mannheim 빌드)로 읽었을 때만 논문 수치와 비교할 수 있습니다.
- `tools/score_docunet_local.py`와 `tools/score_dir300_local.py`는 체크포인트 하나를 예측하고 채점해 `~/.cache/flattener/scores/RUN/`에 보고서를 씁니다.
- 한국어 실사진 평가 세트 KR-Real의 도구와 촬영 안내는 [`tools/krreal/`](tools/krreal/README.md)에 있습니다.
- 채점 규약과 평가기 검증: [인터페이스 사양의 "평가"](docs/INTERFACE_SPEC.md#평가).

## 라이선스

패키지 라이선스는 `Apache-2.0 AND MIT AND OFL-1.1`입니다. 코드는 [Apache-2.0](LICENSE), UVDoc에서 가져온 `flattener.dewarp.model`은 MIT, 함께 배포하는 글꼴은 OFL-1.1입니다.
외부 코드와 데이터의 출처와 이용 조건은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.
