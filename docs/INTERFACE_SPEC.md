# 인터페이스 사양

렌더러 사양 `0.1`. 공개 API는 `flattener`, 좌표 추론은 `flattener.learning`, 페이지 처리는 `flattener.regions`에 있습니다.

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

## 파일과 모델

`render_file(input_path, backward_map, *, target_mask=None, config=None, output_dir=None, overwrite=False)`는 디코딩·렌더링·저장을 잇습니다. `save_result(result, output_dir, *, overwrite=False)`는 이미지·마스크 PNG와 JSON을 쓰고 경로 사전을 돌려줍니다. 덮어쓰기는 명시해야 하며 파일마다 원자적으로 교체합니다. 한 디렉터리에는 작성자 하나만 씁니다.

`load_predictor(path)`는 `learning-0.1` 체크포인트를, `coordinate_study.load_coordinate_predictor(path)`는 기본 모델 `selected.pt`를 포함한 study 체크포인트를 읽습니다. 신뢰할 수 있는 파일만 넘깁니다. 설정·가중치·유한성·체크섬을 검사하고, 로딩 오류는 예외로 알립니다.

`unwarp(image, predictor, output_size, config=...)`는 좌표 예측과 렌더링을 잇습니다. 출력 크기는 예측 격자 이상이어야 합니다. 전체 RGB 입력을 두 축 독립으로 resize해 예측하고 원본에서 샘플링합니다.

좌표·영역 추론 메타데이터의 `processing_status`는 최종 처리 상태이고 `quality_status`는 `"not_evaluated"`입니다. `success`는 기하 검사와 렌더링을 마쳤다는 뜻입니다. `examples/unwarp_photo.py`는 성공하면 0, 실패하면 1로 끝나며 원본 복귀도 1입니다.

## 페이지 영역 API

`PageRegion(left, top, right, bottom)`은 원본 픽셀 경계 사각형입니다. 오른쪽·아래 끝은 exclusive이며 전체 이미지는 `(0,0,width,height)`입니다. 수동 영역은 원본 범위 안의 양의 크기여야 합니다.

`crop_page(image, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 사각형 크롭을, `unwarp_page(image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 영역 조건부 곡면 보정을 합니다.

`region`과 `segmenter` 중 하나만 지정합니다. 분할기는 `load_segmenter(checkpoint)`로 읽고 인쇄 페이지를 검출합니다. `estimate_page_region(image, segmenter, config=...)`는 영역 추정 결과를 돌려줍니다. 자동 분할은 실험 단계이므로 추정 영역을 결과에서 확인합니다.

영역 경로는 미리보기 좌표를 원본 좌표에 합성한 뒤 한 번만 샘플링하고, 추정 마스크가 영역을 충분히 덮지 못하면 영역을 넓혀 다시 검사합니다. 출력 종횡비는 호출자가 정하고, 반환 마스크는 위 렌더러 규약을 따릅니다.

`unwarp_page`의 `predictor`로 `flattener.dewarp`의 `Dewarper`도 받습니다. 이때는 영역에 변마다 크기의 4%를 더한 정수 픽셀 크롭에서 한 번 예측하고, 격자를 원본 좌표에 배치해 `output_size`로 조밀화합니다. 분할기 마스크의 추정 덮는 비율은 검사만 하고(`page_coverage_insufficient`) 영역을 넓히지 않습니다. 수동 사각형의 덮는 비율은 기록만 합니다(`coverage_enforced: false`). `LoadedPredictor` 경로는 그대로입니다.

## 페이지 게이트

`flattener.regions.context`의 `load_context_segmenter(checkpoint, *, device="cpu")`는 `region-context-unet-2.0` 체크포인트를 읽습니다. 파일 옆에 SHA-256 16진수 한 줄을 담은 `<파일>.sha256`이 있어야 합니다. 스키마는 `independent-locator-pilot-1.0`(위치 추정 연구의 `best.pt`)과 `region-context-1`(가중치, 설정, 전처리 버전, 출처 기록)입니다. 전처리는 `regions.segmentation.preprocess_image`이고, 읽은 모델은 `predict_probability`, `model_id`, `model_sha256`을 가지므로 `segmenter=` 자리에 넣을 수 있습니다.

`gate_page_region(probability, source_size, *, config=None)`과 `gate_page(image, segmenter, *, config=None)`는 `detect_region`을 면적 상한 1.0으로 돌려 페이지 상자를 정합니다. 마스크가 화면의 90%를 넘거나 테두리 세 면 이상에 닿으면 `fills_frame=True`, `reasons=("page_fills_frame",)`이고 상자는 이미지 전체입니다. 이 경우는 성공입니다. `page_not_found`, `page_area_rejected`, `ambiguous_page_region`, `uncertain_page_region`은 `model_error` 상태의 `FlattenerError`로 알립니다. `confident`는 dominance ≥0.95이고 평균 확률 ≥0.85일 때 참입니다.

## 스캐너

`flattener.scan.pipeline.Scanner`의 결과 `ScanResult`는 기존 `Status` 값만 씁니다. 스캐너의 판정은 `reason_code` 문자열로 나타내고, 코드마다 상태, 심각도(`info`, `warning`, `failure`), 한/영 안내가 `scan.records.REASONS`에 있습니다. `outcome`은 `rectified`, `rectified_with_warnings`, `perspective`, `original_returned`, `failed`입니다. 대체 결과(원근 보정, 원본 반환)는 실패 상태와 사유를 유지하고 `used_fallback=True`입니다. 렌더러의 검사 사유(`fold_or_degenerate` 등)는 그대로 기록합니다.

스캔 계획 `ScanPlan`의 격자는 원본 기준 flattener 값이고 노드는 가장자리 정렬, 페이지가 똑바로 선 순서입니다. 원근 계획의 homography는 출력 위치 `(u, v, 1)`(u, v ∈ [-1, 1])를 원본 flattener 값의 동차 좌표로 보냅니다. 두 경우 모두 조밀 맵은 원본을 한 번 샘플링합니다. `plan_sha256`은 계획의 정규 JSON sha256이고, 같은 값의 미리보기와 저장 결과는 같은 기하를 씁니다.
