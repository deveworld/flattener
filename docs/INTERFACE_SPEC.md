# 인터페이스 사양

렌더러 사양 `0.1`. 공개 API는 `flattener`, 페이지 처리는 `flattener.regions`, 휨 보정은 `flattener.dewarp`, 스캐너는 `flattener.scan`, 평가는 `flattener.benchmark`에 있습니다.

## 입력과 출력

```python
render(image, backward_map, *, target_mask=None, config=None) -> RenderResult
resize_backward_map(coarse_map, output_size, *, config=None) -> dense_map
```

| 항목 | 계약 |
| --- | --- |
| 이미지 | 방향 보정된 NumPy RGB `uint8 (H_src, W_src, 3)` |
| 좌표 | NumPy `float32 (H_out, W_out, 2)`, 마지막 축 `(x, y)`, 원본 기준 정규화 절대 좌표 |
| 대상 마스크 | NumPy `bool (H_out, W_out)`, 생략하면 전체 `True` |
| 크기 | `(height, width)`, 양의 정수, `bool` 제외 |
| 성공 결과 | RGB `uint8` 이미지와 출력 크기의 `bool` 마스크 3개 |
| 입력 소유권 | 호출자의 배열을 그대로 둠 |

파일 입력은 단일 프레임 8비트 RGB/회색조 JPEG·PNG입니다. `load_image`는 EXIF 방향을 반영하고 회색조를 RGB로 복제합니다. 알파·팔레트·CMYK·다중 프레임 파일은 거부하고, 픽셀 값은 ICC 변환 없이 읽습니다.

## 좌표

출력 `[i, j]`는 `backward_map[i, j]`가 가리키는 원본 위치를 읽습니다. 픽셀 `(0, 0)`은 왼쪽 위 픽셀 중심이며 x는 오른쪽, y는 아래로 커집니다.

```text
g_x = 2 * (x + 0.5) / W_src - 1
g_y = 2 * (y + 0.5) / H_src - 1
x = (g_x + 1) * W_src / 2 - 0.5
y = (g_y + 1) * H_src / 2 - 0.5
I[i, j] = (2 * (j + 0.5) / W_out - 1, 2 * (i + 0.5) / H_out - 1)
```

`identity_map(output_size)`는 기준 격자 I를 만듭니다. 입출력 크기가 같으면 항등 변환, 다르면 half-pixel resize입니다. 정규화 경계 `-1, 1`은 원본 픽셀의 바깥 경계입니다. 변환 함수는 `pixel_to_normalized`, `normalized_to_pixel`입니다.

저해상도 확대는 `I_out + bilinear_resize(coarse_map - I_coarse)`입니다. `align_corners=False`, `antialias=False`이며 residual 가장자리 값을 연장합니다. 유한한 `float32` 맵, 격자 각 축 ≥2, 출력 각 축 ≥격자 크기가 필요합니다. 항등과 상수 이동은 그대로 보존되고, 일반 변형에는 보간 오차가 생깁니다.

`flattener.dewarp`의 격자 `(Gh, Gw, 2)`도 같은 값 규약을 씁니다. 노드는 가장자리 정렬이라 모서리 노드가 페이지 모서리이고, `grid_to_map`이 격자를 위 좌표 맵으로 바꿉니다. UVDoc(`align_corners=True`)과 SyntheticDoc 좌표 변환은 `flattener.dewarp.coords`에 있습니다.

## 샘플링과 마스크

CPU `float32`로 RGB를 [0,1]로 바꾸고 `grid_sample(mode="bilinear", padding_mode="border", align_corners=False)`로 샘플링합니다. 결과 양자화는 `floor(clip(value, 0, 1) * 255 + 0.5)`입니다.

- `source_mask`: 치환 전 좌표가 유한하고 두 축 모두 [-1,1] 안에 있음.
- `target_mask`: 호출자가 대응점이 정의된 보정 대상으로 지정한 위치.
- `valid_mask`: 두 마스크의 교집합. 나머지 픽셀은 `fill_rgb`로 채움.

대상 영역의 NaN·무한대는 실패입니다. 대상 밖의 비유한 값은 허용하고, 샘플링용 복사본에서만 안전한 값으로 치환합니다. 원본 좌표의 이탈량은 검사와 통계에 씁니다.

## 검증과 실패

검사 순서는 입력 shape/dtype/크기 → target 유한성·빈 영역 → 범위 이탈 비율 → 국소 접힘·퇴화 → 유효 픽셀 수입니다. 국소 검사는 target 꼭짓점 4개가 모두 유효한 셀의 bilinear 면을 씁니다. `float64`로 네 코너의 `det(dP/ds, dP/dt) / (4/(W_out*H_out))`를 계산하고 하한 이하이면 실패합니다. 검사는 셀 단위이며, 전역 겹침과 가림은 호출자가 결과에서 확인합니다.

`RenderConfig` 기본값은 입력/출력 각 `16_000_000`픽셀, 한 변 `8192`, target 범위 이탈 상한 `0.25`, Jacobian 비율 하한 `1e-6`, 채움색 `(255,255,255)`, 실패 정책 `"raise"`입니다. 검사할 셀이 없으면 `fold_check="not_applicable"`, 검사 전에 멈추면 `null`입니다.

`RenderResult`는 `status`, `reason_code`, `message`, `image`, 마스크 3개, `used_fallback`, `metadata`를 가집니다. 상태는 `success`, `input_error`, `invalid_transform`, `model_error`, `render_error`, `io_error`입니다.

- `raise`: 상태·사유·메타데이터를 담은 `FlattenerError`를 던짐.
- `return`: 실패 결과의 이미지와 마스크는 `None`.
- `original`: 검증된 방향 보정 원본의 복사본과 `used_fallback=True`. 실패 상태는 유지하고 마스크는 `None`. 입력 검증이 실패하면 `return` 정책을 적용.

잘못된 실패 정책과 배열 반환 helper의 오류는 예외입니다. fallback 크기는 원본 크기입니다. 메타데이터에는 설정·크기·모델 해시·버전·검사 통계·실패 단계·단계별 시간을 기록합니다.

## 파일

`render_file(input_path, backward_map, *, target_mask=None, config=None, output_dir=None, overwrite=False)`는 디코딩·렌더링·저장을 잇습니다. `save_result(result, output_dir, *, overwrite=False)`는 이미지·마스크 PNG와 JSON을 쓰고 경로 사전을 돌려줍니다. 덮어쓰기는 명시해야 하며 파일마다 원자적으로 교체합니다. 한 디렉터리에는 작성자 하나만 씁니다.

영역 추론 메타데이터의 `processing_status`는 최종 처리 상태이고 `quality_status`는 `"not_evaluated"`입니다. `success`는 기하 검사와 렌더링을 마쳤다는 뜻입니다.

## 페이지 영역 API

`PageRegion(left, top, right, bottom)`은 원본 픽셀 경계 사각형입니다. 오른쪽·아래 끝은 exclusive이며 전체 이미지는 `(0,0,width,height)`입니다. 수동 영역은 원본 범위 안의 양의 크기여야 합니다.

`crop_page(image, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 사각형 크롭을, `unwarp_page(image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 영역 조건부 곡면 보정을 합니다.

`region`과 `segmenter` 중 하나만 지정합니다. 분할기는 `flattener.regions.context.load_context_segmenter(checkpoint)`로 읽은 페이지 게이트이고 인쇄 페이지를 검출합니다. `estimate_page_region(image, segmenter, config=...)`는 영역 추정 결과를 돌려줍니다. 자동 분할은 실험 단계이므로 추정 영역을 결과에서 확인합니다.

두 함수 모두 원본을 한 번만 샘플링합니다. `crop_page`는 추정 마스크가 영역을 충분히 덮지 못하면 영역을 넓혀 다시 검사합니다. 출력 종횡비는 호출자가 정하고, 반환 마스크는 위 렌더러 규약을 따릅니다.

`unwarp_page`의 `predictor`는 `flattener.dewarp`의 `Dewarper`입니다. 다른 값은 `invalid_predictor`로 거부합니다. 영역에 변마다 크기의 4%를 더한 정수 픽셀 크롭에서 한 번 예측하고, 격자를 원본 좌표에 배치해 `output_size`로 조밀화합니다. 분할기 마스크의 추정 덮는 비율은 검사만 하고(`page_coverage_insufficient`) 영역을 넓히지 않습니다. 수동 사각형의 덮는 비율은 기록만 합니다(`coverage_enforced: false`).

## 페이지 게이트

`flattener.regions.context`의 `load_context_segmenter(checkpoint, *, device="cpu")`는 `region-context-unet-2.0` 체크포인트를 읽습니다. 파일 옆에 SHA-256 16진수 한 줄을 담은 `<파일>.sha256`이 있어야 합니다. 스키마는 `independent-locator-pilot-1.0`(위치 추정 연구의 `best.pt`)과 `region-context-1`(가중치, 설정, 전처리 버전, 출처 기록)입니다. 전처리는 `regions.segmentation.preprocess_image`이고, 읽은 모델은 `predict_probability`, `model_id`, `model_sha256`을 가지므로 `segmenter=` 자리에 넣을 수 있습니다.

`gate_page_region(probability, source_size, *, config=None)`과 `gate_page(image, segmenter, *, config=None)`는 `detect_region`을 면적 상한 1.0으로 돌려 페이지 상자를 정합니다. 마스크가 화면의 90%를 넘거나 테두리 세 면 이상에 닿으면 `fills_frame=True`, `reasons=("page_fills_frame",)`이고 상자는 이미지 전체입니다. 이 경우는 성공입니다. `page_not_found`, `page_area_rejected`, `ambiguous_page_region`, `uncertain_page_region`은 `model_error` 상태의 `FlattenerError`로 알립니다. `confident`는 dominance ≥0.95이고 평균 확률 ≥0.85일 때 참입니다.

v0 가중치는 위치 추정 연구의 `best.pt`(파라미터 140,545개, 입력 192×144)입니다. 처음부터 8 epoch 학습했고, 학습 자료는 자체 합성 장면 3,840장, SmartDoc 원본 문서로 만든 휨 합성 1,584장, SmartDoc 촬영 사진 528장입니다.

- `tools/locator_v0.py install`은 `outputs/models/locator.zip`의 `best.pt`를 `locator-v0-pilot.pt`로 복사하고, 가중치와 출처 기록(`tools/locator_v0_provenance.json`)을 담은 `page-gate-v0.pt`를 씁니다. 두 파일에 `.sha256` 파일이 붙습니다. 결과의 `matches_recorded`는 export가 기록된 sha256과 같은지 알려 줍니다.
- 출처 기록은 연구 때의 데이터 manifest에서 뽑아 고정했습니다. 그 안의 경로(`outputs/dataset`, `outputs/coordinate-fold/locator.zip`)는 연구 당시 위치입니다. 이 문자열도 export의 sha256에 들어가므로 바꾸지 않습니다. `locator_v0_smartdoc.json`의 경로도 같습니다.
- `reproduce`는 SmartDoc 촬영 검증 샘플 48장(`outputs/benchmarks/smartdoc/manifest.json`의 `gate_v0`, 파일은 `gate-v0/`)을 연구와 같은 방식(시그모이드, 384×288로 쌍선형 확대, 임계값 0.5)으로 잽니다. CPU에서 IoU는 기록값 0.9508340768516064와 같고, 이미지별 손실 차이는 2×10⁻⁷ 이하, 보관된 모델 코드와의 확률 차이는 0입니다.
- `src/flattener/regions/locator_v0_smartdoc.json`은 v0가 학습에 쓴 SmartDoc 문서 22종(`exclude_from_evaluation`), 체크포인트 선택에 쓴 4종(`selection_exposed`), 쓰지 않은 4종(`unexposed`)입니다. v0를 SmartDoc으로 평가할 때는 앞의 22종을 배경과 관계없이 모두 빼고, 선택에 쓴 4종은 따로 보고합니다. `context.smartdoc_exposure()`로 읽습니다.
- 페이지가 없는 사진으로는 평가한 적이 없습니다.

## 스캐너

`flattener.scan.pipeline.Scanner`의 결과 `ScanResult`는 기존 `Status` 값만 씁니다. 스캐너의 판정은 `reason_code` 문자열로 나타내고, 코드마다 상태, 심각도(`info`, `warning`, `failure`), 한/영 안내가 `scan.records.REASONS`에 있습니다. `outcome`은 `rectified`, `rectified_with_warnings`, `perspective`, `original_returned`, `failed`입니다. 대체 결과(원근 보정, 원본 반환)는 실패 상태와 사유를 유지하고 `used_fallback=True`입니다. 렌더러의 검사 사유(`fold_or_degenerate` 등)는 그대로 기록합니다.

스캔 계획 `ScanPlan`의 격자는 원본 기준 flattener 값이고 노드는 가장자리 정렬, 페이지가 똑바로 선 순서입니다. 원근 계획의 homography는 출력 위치 `(u, v, 1)`(u, v ∈ [-1, 1])를 원본 flattener 값의 동차 좌표로 보냅니다. `--inset` e와 `--border` b는 두 계획에 똑같이 적용되어, 출력 u는 페이지 위치 (1 − e + b)·u를 보여 줍니다. 두 경우 모두 조밀 맵은 원본을 한 번 샘플링합니다. `plan_sha256`은 계획의 정규 JSON sha256이고, 같은 값의 미리보기와 저장 결과는 같은 기하를 씁니다.

단계는 read(EXIF 1회, 16MP를 넘는 JPEG는 draft decode) → detect(페이지 게이트, 필요할 때만 사진 전체를 한 번 추론한 결과와의 합의 규칙) → orient(OrientNet) → infer(첫 추론과 크롭 재추론 1회, deskew. 크롭 재추론이 페이지를 4% 넘게 옮기면 한 번 더) → verify(평면 미리보기 방향 확인, 게이트 영역과 비교, 작은 평면 미리보기의 종이·글줄 검사, 패스 사이 노드 이동 ≤ 페이지 대각선의 4%) → plan(출력 크기 auto) → render(`flattener.render` 1회) → save입니다.

- 페이지 영역은 휨 보정 모델이 찾은 윤곽이 기준이고, 페이지 게이트는 참고로 씁니다. 게이트는 큰 사진에서 첫 추론의 크롭을 정하고, 페이지가 없는 사진을 거르고, 후보를 고르는 데 씁니다. 게이트 영역이 편 페이지보다 넓으면(옆 쪽, 책 옆면, 밑에 깔린 종이) 결과를 그대로 내고 `page_region_larger` 경고를 남깁니다. 이때 편 결과는 종이처럼 보여야 합니다. 긴 변 384픽셀 미리보기에서 채도 중앙값이 0.28 이하이고 어두운 화소(밝기 0.35 미만)가 35% 이하여야 하고, 아니면 페이지가 없는 사진(노트북 키보드, 나무 책상)으로 보고 `page_not_found`로 원본을 돌려줍니다. 또 첫 크롭 상자의 한 변(사진 가장자리가 아닌 변)을 따라 페이지와 같은 색의 빈 종이가 절반 넘게 이어지면 크롭이 페이지를 자른 것일 수 있습니다(게이트가 흰 여백을 놓친 경우). 이때 사진 전체를 한 번 추론한 윤곽이 게이트 영역을 0.95 이상 담고, 편 페이지에 더하는 넓이가 페이지 넓이 이하이며 그 절반 이상이 페이지 색이면 이 후보를 실패로 두고(`unstable_geometry`, `page_continues`) 사진 전체 윤곽 후보로 넘어갑니다. 페이지 색은 윤곽 안 글자 없는 화소의 중앙값이고, 채널마다 0.08 안이면 같은 색으로 봅니다. 게이트 영역이 페이지 네 변 모두에서 페이지 넓이의 5%를 넘게 남으면 페이지 일부만 편 것으로 보고 실패로 처리합니다. 수동 사각형과 비교할 때는 덮는 비율 0.95 미만이 실패입니다. 게이트가 영역을 여러 조각으로 나누면 사진 전체를 한 번 추론한 윤곽을 페이지로 쓰고 `ambiguous_page_region`이나 `two_page_candidate`를 경고로 남깁니다. 윤곽이 어느 조각도 담지 못하거나, 담은 조각이 윤곽 넓이의 절반을 덮지 못하면 원본을 돌려줍니다.
- 편 결과가 페이지 한 조각을 크게 늘린 것이면 페이지가 사진 밖으로 잘린 근접 사진으로 보고 원본을 돌려줍니다(`page_cut_off`, "페이지 끝이 사진 밖으로 잘렸어요. 조금 떨어져서 페이지 전체가 보이게 다시 찍어 주세요"). 긴 변 384픽셀 미리보기에서 글줄이 페이지를 가로질러 8줄 이상 20줄 미만이고(간격이 페이지의 1/8보다 길면 단이나 표 같은 배치로 봅니다), 글줄 간격이 규칙적이고(자기상관 0.4 이상), 두 변 이상에서 가장자리 띠의 절반 넘게 글자가 걸리면 조각으로 봅니다. 이때는 다른 후보도 같은 사진을 보므로 후보 시도를 멈추고, `--fallback perspective`여도 원근 보정을 하지 않습니다. 수동 영역과 수동 사각형에는 이 검사를 하지 않으므로 원하면 직접 영역을 지정해 펼 수 있습니다.
- 후보는 기본 경로, deskew 없이, 다른 위치 추정기가 찾은 영역, 사진 전체 1패스(`rectify`) 순서로 시도하고 먼저 통과한 것을 씁니다. 기록에는 이 순서대로 `default`, `no_deskew`, `other_box`, `whole_photo`로 남습니다. 마지막 후보도 방향 k만큼 격자 노드를 재색인해 페이지를 바로 세웁니다. 기본 경로가 실패하고 뒤 후보가 통과하면 `candidate_fallback` 경고가 남습니다. 크롭 재추론만 실패하면 첫 추론 결과를 씁니다(`rerun_discarded`). `--fallback perspective`의 원근 보정 결과에는 `curvature_not_corrected`가 붙습니다.
- 두 쪽 나누기(`ScanOptions.pages`, 기본 `auto`): 받아들인 계획의 평면 미리보기가 가로로 넓고(너비/높이 1.0 이상) 가운데 띠(너비의 35–65%)에 글자가 없는 세로 줄이 있으며 양쪽 글자가 그 줄보다 2.5배 이상 많으면 펼친 두 쪽으로 봅니다(`look.fold`). 격자를 접힌 곳에서 둘로 나눠 쪽마다 크기를 정하고 검사하고 기록합니다(사유 `two_pages`, 기록의 `metadata.page`). 한 쪽만 검사에 실패하면 그 쪽만 대체 결과가 됩니다. `pages=2`는 접힌 곳을 못 찾으면 가운데에서 나누고, `pages=1`은 나누지 않습니다. 원근 보정 계획과 `uvdoc` 노드 격자는 나누지 않습니다.
- 수동 수정(`--region`, `--quad`, `--quarter`, `--no-dewarp`, `--long-side`, `--output-size`, `--aspect`)은 자동 경로와 같은 단계를 거치고 기록에 `manual`로 남습니다. `--no-dewarp`도 방향을 지정하지 않으면 OrientNet에 묻습니다. 사각형은 `--quad`, `--region`, 페이지 게이트의 영역 순으로 정하고, 게이트가 없으면 사진 전체를 한 번 추론한 윤곽의 네 꼭짓점을 씁니다. `--quad`는 왼쪽 위부터 시계 방향으로 줍니다. 반시계 방향은 뒤집힌 페이지를 뜻하므로 입력 오류입니다. `--no-dewarp`로 요청한 원근 보정 결과는 스캔된 것으로 셉니다. 모델 출력은 입력 sha256별로 캐시하므로 바뀐 단계 뒤만 다시 계산합니다.
- 판정 임계값(`scan.locate.THRESHOLDS`, `scan.pipeline.VERIFY`)은 설계의 시작값이고 스캔 기록에 남습니다. SmartDoc train 분할과 합성 val로 정하는 일이 남아 있습니다. 종이·글줄·크롭 검사의 값과 조각 나눔의 0.5는 KR-Real dev, SmartDoc train, 페이지가 없는 사진 76장(배경 크롭과 질감 사진)으로 정했습니다.
- `--timing F.jsonl`은 사진마다 단계별 시간(디코드 포함, 패스별 `infer_pass_*`, `render_map`/`render_check`/`render_sample`, `preview`), 저장까지의 `end_to_end`, 캐시 적중을 한 줄씩 쓰고, 첫 줄에 모델 로드 시간을 남깁니다. 같은 사진을 반복해 잴 때는 `--no-cache`를 씁니다.
- `--fast-check`(`ScanOptions(fast_check=True)`)는 렌더러의 맵 검사 판정을 격자에서 계산하고, 작은 proxy에서만 `flattener.render`를 돌립니다. 저장 이미지는 같습니다. 방법은 `scan/fast_check.py`에, 판정 비교는 `tools/fast_check_agreement.py`에 있습니다.
- `rectify_batch(scanner.dewarper, inputs, out, **scanner.batch_kwargs(options))`는 벤치마크 예측 폴더를 스캐너로 만듭니다. `predictions.json`에 pipeline과 스캐너의 양자화·inset·pass 설정이 남고, 다른 pipeline의 예측이 있는 폴더나 스캐너 설정과 다른 인자는 거부합니다.

## 휨 보정 예측

`python -m flattener.dewarp.predict --weights CKPT {image,docunet}`의 예측은 `flattener.render`의 검사(좌표 범위, 덮는 비율, 접힘)를 거칩니다. 검사에 실패한 사진은 저장하지 않고 상태(`invalid_transform`과 이유)를 출력과 `predictions.json`에 남기며, 명령은 종료 코드 1로 끝납니다. `--force-render`는 실패한 사진도 저장합니다. 공개 방법들과 같은 방식으로 벤치마크를 채점할 때 씁니다. 이때도 실패 기록과 종료 코드 1은 그대로이고, 보고서의 `prediction_failures`가 실패한 사진 수를 셉니다. `--weights`에는 UVDoc 공개 가중치 `best_model.pkl`도 넣을 수 있습니다.

- `--quantize truncate`는 샘플링한 값을 `floor(x*255)`로 저장합니다(기본값 `round`). UVDoc·SyntheticDoc 논문 수치는 이렇게 저장한 결과를 UVDoc 스크립트 방식(`--ref-resize uvdoc_view2_double`)으로 채점한 값입니다. 반올림과 내림만 바꿔도 LD가 0.3 정도 달라집니다.
- `--inset`은 렌더링 전에 예측한 페이지 윤곽을 안쪽으로 옮깁니다. 출력 좌표 u(−1…1)가 예측 격자의 (1−e)·u 위치를 보여 주므로 각 변이 페이지 크기의 e/2만큼 안으로 들어옵니다. 값은 프리셋 이름, `E`, `EX,EY`입니다. 프리셋 `syntheticdoc_half_cell`(x 1/177, y 1/121)은 SyntheticDoc 추론이 121×177 맵을 `align_corners=True`로 키울 때 생기는 반 칸 이동입니다.
- `--rerun R`은 사진 전체로 한 번 예측한 뒤 R번 더 잘라서 다시 예측합니다. 현재 맵의 페이지 윤곽에 축을 맞춘 상자를 잡고 변마다 4%를 더해 정수 픽셀로 자른 뒤, 새 격자를 크롭 위치와 합성해 원본 좌표로 되돌립니다. `--deskew`는 위아래 변 방향으로 페이지 각도를 추정해 45° 이하이면 똑바로 돌린 뒤 예측합니다. 건너뛴 이유(`degenerate_outline`, `inconsistent_edges`, `angle_above_limit`)와 상자, 각도는 `crops`에 남습니다.
- 양자화, inset, rerun 설정은 `predictions.json`과 `evidence`에 남고, 다른 설정으로 만든 예측은 다시 쓰지 않습니다. `uvdoc-predict`, `dir300-predict`, `docunet-stress`, `tools/score_docunet_local.py`도 같은 옵션을 받습니다.
- 학습 검증은 화면 종류별로 EPE를 따로 기록합니다. 화면은 페이지에 꼭 맞춘 크롭(`page`), DocUNet처럼 여백을 둔 크롭(`docunet`), 사진 전체(`full_frame`), ±90° 돌린 크롭(`rot90`, `rot270`), 무늬 캔버스의 10–25%만 차지하는 작은 페이지(`small_page`)입니다. 최고 체크포인트는 `val_select`로 정한 화면 하나로 고릅니다(기본값 `page`). 유효한 정답 점이 없는 샘플은 `not_evaluable`로 셉니다.

## 평가

`flattener.benchmark`는 DocUNet의 MS-SSIM·LD·AD를 공개 MATLAB 스크립트와 같은 방식으로, OCR의 ED·CER을 pytesseract 호출과 같은 방식으로 계산합니다.

- DocUNet 입력은 크롭 사진 `outputs/benchmarks/docunet/crop/`, 스캔 `reference/docunet-reference-{k}.png`, 원본 사진 `photo/`(스트레스 평가의 `full_frame`)입니다. 파일별 해시와 출처는 같은 폴더의 `manifest.json`에 있습니다.
- `--convention`은 `130_raw`(PaperEdge, DocTr 계열), `128_excl64`(UVDoc, 64번 문서를 뺀 128장), `130_rot64` 중에서 고르고 기본값은 셋 모두입니다. `--ref-resize uvdoc_view2_double`은 UVDoc 스크립트가 view 2에서 스캔을 두 번 리사이즈하는 동작을 재현합니다. 논문과 비교할 때는 8비트 내림(`--quantize truncate`), `128_excl64`, `uvdoc_view2_double`을 씁니다.
- AD는 MATLAB `lsqr` 기본값을 따르는 `lsqr_4iter`로 보고하고, 정확한 최소제곱 값 `exact_lstsq`도 함께 저장합니다.
- LD·AD에는 SIFT flow C++ 소스(Ce Liu)가 필요합니다. DocUNet `eval.zip`이나 UVDoc `eval_code`의 `SIFTflow` 폴더를 `--siftflow-source` 또는 `FLATTENER_SIFTFLOW_SOURCE`로 지정하면 처음 실행할 때 C++ 컴파일러와 ninja로 빌드합니다(`build-siftflow`).
- 평가기 보정 검사는 네 가지입니다. PaperEdge 결과 이미지로 PaperEdge가 공개한 MATLAB 값을 재현하는지(`paperedge_matlab`), 같은 이미지로 UVDoc 논문 표의 PaperEdge 행을(`uvdoc_table_paperedge`), UVDoc 공개 결과로 같은 표의 UVDoc 행을(`uvdoc_table_uvdoc`), Tesseract 5.0.1로 DocScanner OCR 표를(`ocr_table`) 재현하는지 봅니다. 결과는 `outputs/benchmarks/docunet/calibration.json`에 있습니다.
- OCR 보고서의 `comparable`은 OCR 점수 재현 검사를 통과한 Tesseract 빌드로 읽고 스캔 해시가 맞을 때만 `true`입니다. 빌드 목록은 `flattener.benchmark.ocr.CALIBRATED_TESSERACT_BUILDS`이고, 지금은 UB-Mannheim Windows 빌드 `v5.0.1.20220118`(Leptonica 1.78.0, tessdata_fast `eng`) 하나입니다. Linux에서는 설치 파일을 `7z x`로 풀고 Wine으로 실행합니다(`--tesseract TREE/tesseract.exe --launcher wine`, `WINEPREFIX`와 `WINEDEBUG=-all`은 환경 변수). conda-forge `tesseract=5.0.1`(Leptonica 1.82.0)은 재현 검사를 통과하지 못합니다.

스트레스 평가(`docunet-stress --weights CKPT --output-dir OUT --force-render`)는 DocUNet 사진을 바꿔 가며 같은 평가기로 채점합니다. 변형은 `crop`(기준), `full_frame`(배경까지 담긴 원본 사진, EXIF 방향 적용), `rot90`·`rot270`(크롭을 ±90° 돌림), `small_page`(무늬 캔버스 면적의 10–25%)이고, 출력은 똑바로 선 크롭의 크기로 렌더링합니다. 스캔은 똑바로 둔 채 채점하므로 돌아간 페이지를 그대로 돌려주는 모델은 `rot90`·`rot270` 점수가 낮습니다. 결과는 `OUT/stress.json`과 변형별 `OUT/<변형>/docunet.json`에 남습니다. UVDoc 공개 가중치의 기준선은 `outputs/benchmarks/docunet/stress/uvdoc-mit/`에 있습니다.

UVDoc 벤치마크(실사진 50장)는 UVDoc 공개 코드(`uvdocBenchmark_pred.py`, `evalScriptUVDoc.m`, MIT)의 규약을 따릅니다.

- `uvdoc-prepare UVDoc_benchmark.zip`은 압축 파일(https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip)의 SHA-256을 확인하고 `outputs/benchmarks/uvdoc-bench/`에 푼 뒤 `manifest.json`을 씁니다.
- 휨 보정 학습 자료의 폴더가 `outputs/benchmarks/` 안에 있거나, 그 폴더나 상위 폴더에 벤치마크 `manifest.json`이 있으면 학습은 `benchmark_data` 오류로 멈춥니다.
- `uvdoc-predict`는 사진 `img/{id}.png`로 격자를 예측하고, 그 격자를 휘어 놓은 텍스처 `warped_textures/{id}.png`에 적용해 `OUT/uwp_texture/`에 씁니다. 쌍선형 보간, 원본 바깥은 0, `floor(x*255)` 저장까지 공개 코드와 같습니다.
- `uvdoc`은 `uwp_texture`를 평평한 텍스처 `texture_sample/{id}.png`와 DocUNet과 같은 MATLAB 에뮬레이션으로 비교하고 50장 평균을 냅니다. 비교 기준은 UVDoc README erratum의 50장 값(MS-SSIM 0.785, AD 0.119)입니다.
- 평가기 검증: UVDoc 공개 결과 이미지는 MS-SSIM 0.7854, AD 0.1186, LD 6.812로 erratum 값과 AADD 표 S1의 LD 6.810에 맞습니다.

DIR300(실사진 300장, OCR 90장)은 DocGeoNet의 공개 규약(`ssim_ld_eval_DIR300.m`, `OCR_eval_DIR300.py`)을 보고 직접 구현했습니다.

- 자료는 `outputs/benchmarks/dir300/`의 `dist300/`(입력 사진)과 `gt300/`(기준 이미지)이고 비상업 평가 전용입니다. 두 폴더는 공식 파일과 해시가 같아야 합니다.
- `dir300-predict`는 사진마다 원래 크기로 보정합니다. 가장 큰 사진(3472×4624) 때문에 렌더러 한도를 2,000만 화소로 둡니다.
- `dir300`은 기준 이미지를 598,400화소로, 보정 결과를 그 크기로 리사이즈해 MS-SSIM·LD(DocUNet `evalUnwarp.m`)와 AD(PaperEdge `evalAlignedUnwarp.m`)를 재고 300쌍의 단순 평균을 냅니다. `dir300-ocr`은 DocGeoNet 목록의 90장을 pytesseract 기본값으로 읽습니다.
- DIR300 사진에는 배경이 함께 찍혀 있어, 사진 전체를 한 번에 펴는 모델은 페이지를 먼저 찾는 방법보다 점수가 크게 낮습니다. 논문 표에는 두 방식이 섞여 있습니다.
- 평가기 검증: UVDoc MIT 가중치를 `--quantize truncate`로 예측하면 0.6215 / 7.742 / 0.2183으로 TADoc 표의 0.622 / 7.73 / 0.218과 맞습니다.

`tools/score_docunet_local.py CKPT RUN`과 `tools/score_dir300_local.py CKPT RUN`은 예측과 채점을 한 번에 하고, 보고서를 `$FLATTENER_CACHE_DIR/scores/RUN/`(기본 `~/.cache/flattener/scores/RUN/`)에 설정별 파일 이름(예: `docunet-uvdoc_view2_double-truncate.json`)으로 씁니다. 모델별 점수 모음은 `outputs/models/scores.json`에 있습니다.

## 결과의 근거

`predictions.json`, 학습 로그·`status.json`·`best.json`, `flattener.benchmark` 보고서에는 `evidence` 항목이 들어갑니다. `evidence_kind`로 결과의 종류를 구분합니다.

| `evidence_kind` | 뜻 |
| --- | --- |
| `new_model_real_photo` | 직접 학습한 체크포인트(1만 스텝 이상)로 DocUNet, UVDoc 벤치마크, DIR300 실사진을 보정하고 채점한 결과. 새 모델의 실사진 성능으로 세는 것은 이것 하나입니다 |
| `gt_crop_validation`, `gt_full_frame_validation` | 학습 중 검증. 정답이 있는 샘플을 정답 페이지 영역에 맞춰 자르거나 전체 화면으로 넣어 잽니다 |
| `external_pretrained` | UVDoc 공개 가중치나 저자가 공개한 결과처럼 밖에서 학습한 모델의 결과. 비교 기준입니다 |
| `smoke_training` | 1만 스텝 미만으로 학습한 체크포인트의 결과 |
| `docunet_standard`, `docunet_stress`, `uvdoc_benchmark`, `dir300_standard` | 예측 폴더에 `predictions.json`이 없어 모델 출처를 모르거나, 첫 연구의 모델(`own_previous_model`)을 채점한 결과 |
| `evaluator_calibration`, `predictions`, `training_loss` | 평가기 보정 검사 보고서, 점수 없는 예측 기록, 학습 손실 |

`measurement`(무엇을 쟀는지), `model_provenance`(`own_checkpoint`, `own_previous_model`, `external_pretrained`), `checkpoint_step`, `initialization`(`scratch` 또는 UVDoc 가중치인 `external_pretrained`)도 함께 기록합니다. UVDoc 가중치에서 시작한 미세조정도 `new_model_real_photo`로 세고, 향상 폭은 같은 변형의 UVDoc 기준선과 비교해 읽습니다.
