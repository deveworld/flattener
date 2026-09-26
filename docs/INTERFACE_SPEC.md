# 인터페이스 사양

렌더러 사양 `0.1`. 공개 API는 `flattener`, 좌표 추론은 `flattener.learning`, 페이지 처리는 `flattener.regions`에서 제공합니다.

## 입력과 출력

```python
render(image, backward_map, *, target_mask=None, config=None) -> RenderResult
resize_backward_map(coarse_map, output_size, *, config=None) -> dense_map
```

| 항목 | 계약 |
| --- | --- |
| 이미지 | 방향 보정된 NumPy RGB `uint8 (H_src, W_src, 3)` |
| 좌표 | NumPy `float32 (H_out, W_out, 2)`, 마지막 축 `(x, y)`, 원본 기준 정규화 절대 좌표 |
| 대상 마스크 | NumPy `bool (H_out, W_out)`, 생략 시 전체 `True` |
| 크기 | `(height, width)`, 양의 정수, `bool` 제외 |
| 성공 결과 | RGB `uint8` 이미지와 출력 크기의 `bool` 마스크 3개 |
| 입력 소유권 | 호출자의 배열 원형 유지 |

파일 입력은 단일 프레임 8비트 RGB/회색조 JPEG·PNG입니다. `load_image`는 EXIF 방향을 반영하고 회색조를 RGB로 복제합니다. 알파·팔레트·CMYK·다중 프레임·ICC 색 변환은 미지원입니다.

## 좌표

출력 `[i, j]`는 `backward_map[i, j]`의 원본 위치를 읽습니다. 픽셀 `(0, 0)`은 왼쪽 위 픽셀 중심이며 x는 오른쪽, y는 아래로 증가합니다.

```text
g_x = 2 * (x + 0.5) / W_src - 1
g_y = 2 * (y + 0.5) / H_src - 1
x = (g_x + 1) * W_src / 2 - 0.5
y = (g_y + 1) * H_src / 2 - 0.5
I[i, j] = (2 * (j + 0.5) / W_out - 1, 2 * (i + 0.5) / H_out - 1)
```

`identity_map(output_size)`는 기준 격자 I를 생성합니다. 동일 입출력 크기는 항등 변환, 다른 크기는 half-pixel resize입니다. 정규화 경계 `-1, 1`은 원본 픽셀의 바깥 경계입니다. 좌표 변환 함수는 `pixel_to_normalized`, `normalized_to_pixel`입니다.

저해상도 확대는 `I_out + bilinear_resize(coarse_map - I_coarse)`입니다. `align_corners=False`, `antialias=False`이며 residual 가장자리 값을 연장합니다. 유한한 `float32` 지도, 격자 각 축 ≥2, 출력 각 축 ≥격자 크기가 필요합니다. 항등·상수 이동을 보존하고 일반 변형에는 보간 오차가 생깁니다.

## 샘플링과 마스크

CPU `float32`로 RGB를 [0,1]에 변환하고 `grid_sample(mode="bilinear", padding_mode="border", align_corners=False)`로 샘플링합니다. 결과 양자화는 `floor(clip(value, 0, 1) * 255 + 0.5)`입니다.

- `source_mask`: 치환 전 좌표가 유한하고 두 축 모두 [-1,1] 범위에 포함.
- `target_mask`: 호출자가 대응점이 정의된 보정 대상으로 지정한 위치.
- `valid_mask`: 두 마스크의 교집합. 나머지 픽셀은 `fill_rgb`로 채움.

대상 영역의 NaN·무한대는 실패입니다. 대상 밖의 비유한 값은 허용하며 샘플링용 복사본에서 안전한 값으로 치환합니다. 원본 좌표의 이탈량을 검사·통계에 사용합니다.

## 검증과 실패

검사 순서는 입력 shape/dtype/크기 → target 유한성·빈 영역 → 범위 이탈 비율 → 국소 접힘·퇴화 → 유효 픽셀 수입니다. 국소 검사는 target 꼭짓점 4개가 유효한 셀의 bilinear 면을 사용합니다. `float64`로 네 코너의 `det(dP/ds, dP/dt) / (4/(W_out*H_out))`를 계산하고 하한 이하이면 실패합니다. 검사 범위는 국소 셀이며 전역 겹침·가림 검증은 미지원입니다.

`RenderConfig` 기본값은 입력/출력 각 `16_000_000`픽셀, 한 변 `8192`, target 범위 이탈 상한 `0.25`, Jacobian 비율 하한 `1e-6`, 채움색 `(255,255,255)`, 실패 정책 `"raise"`입니다. 검사 가능한 셀이 없으면 `fold_check="not_applicable"`, 미실행이면 `null`입니다.

`RenderResult`는 `status`, `reason_code`, `message`, `image`, 마스크 3개, `used_fallback`, `metadata`를 가집니다. 상태는 `success`, `input_error`, `invalid_transform`, `model_error`, `render_error`, `io_error`입니다.

- `raise`: 상태·사유·메타데이터를 담은 `FlattenerError`.
- `return`: 실패 결과의 이미지와 마스크는 `None`.
- `original`: 검증된 방향 보정 원본의 복사본과 `used_fallback=True`. 실패 상태 유지, 마스크는 `None`. 입력 검증 실패 시 `return` 정책 적용.

잘못된 실패 정책과 배열 반환 helper 오류는 예외입니다. fallback 크기는 원본 크기입니다. 메타데이터는 설정·크기·모델 해시·버전·검사 통계·실패 단계·시간을 담습니다. 미실행 시간은 `null`, 적재 모델 재사용 시 `load_model=0`이며 실제 적재 시간은 따로 기록합니다.

## 파일과 모델

`render_file(input_path, backward_map, *, target_mask=None, config=None, output_dir=None, overwrite=False)`는 디코딩·렌더링·저장을 연결합니다. `save_result(result, output_dir, *, overwrite=False)`는 이미지·마스크 PNG와 JSON, 경로 사전을 제공합니다. 덮어쓰기는 명시적으로 허용하며 파일별 원자적 교체를 사용합니다. 같은 디렉터리는 단일 작성자로 운영합니다.

`load_predictor(path)`는 기존 좌표 가중치를, `coordinate_study.load_coordinate_predictor(path)`는 새 study 가중치를 읽습니다. 신뢰할 수 있는 파일을 사용하세요. 설정·가중치·유한성·체크섬을 검사하고 로딩 오류는 예외로 반환합니다.

`unwarp(image, predictor, output_size, config=...)`는 좌표 예측과 렌더링을 연결합니다. 출력 크기를 지정하며 예측 격자 이상의 크기가 필요합니다. 전체 RGB 입력을 두 축 독립 resize하여 예측하고 원본에서 샘플링합니다. 모델은 실험용이며 결과의 글자 경계와 잔여 왜곡을 확인해야 합니다.

좌표·영역 추론 메타데이터의 `processing_status`는 최종 처리 상태, `quality_status`는 `"not_evaluated"`입니다. `success`는 기하 검사와 렌더링 완료를 뜻합니다. `examples/unwarp_photo.py`의 종료 코드는 성공 0, 실패 1이며 원본 복귀에도 실패 코드를 유지합니다.

## 페이지 영역 API

`PageRegion(left, top, right, bottom)`은 원본 픽셀 경계 사각형입니다. 오른쪽·아래 끝은 exclusive이며 전체 이미지는 `(0,0,width,height)`입니다. 수동 영역은 원본 범위 안의 양의 크기를 사용합니다.

`crop_page(image, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 사각형 크롭을 수행합니다. `unwarp_page(image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None)`는 영역 조건부 곡면 보정을 수행합니다.

`region` 또는 `segmenter` 중 하나를 지정합니다. 분할기는 `load_region_model(checkpoint)`로 읽고 인쇄 페이지를 검출합니다. `estimate_page_region(image, segmenter, config=...)`는 영역 추정 결과를 반환합니다. 자동 분할 품질은 기준 미달입니다.

영역 경로는 미리보기 좌표를 원본에 합성한 뒤 한 번 샘플링합니다. 기본 추정 마스크 coverage는 98%이며 부족하면 최대 20%까지 확대 후 재검사합니다. 반환 마스크의 정의는 위 렌더러 규약을 따릅니다. 출력 종횡비는 호출자가 정하고 실제 글자 보존은 결과 이미지에서 확인합니다.
