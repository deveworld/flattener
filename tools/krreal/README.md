# KR-Real v1 파일럿 키트

한국어 실사진 평가 세트 KR-Real v1의 첫 60장을 찍고, 라벨을 달고, 전사하고, 목록(manifest)을 만드는 도구입니다. 사진은 비공개 평가 자료입니다.

- 사진, 라벨, 전사, manifest는 모두 `outputs/krreal/v1/`(git이 무시하는 폴더)에 둡니다. 키트의 명령은 git이 추적하는 폴더를 거부합니다.
- 사진은 학습, 체크포인트 선택에 쓰지 않습니다. dev 분할(출처의 40%)은 임계값 조정에, test 분할(60%)은 마일스톤마다 한 번 보고에 씁니다.
- 공개할 수 있는 것은 퍼블릭 도메인 인쇄물을 찍은 사진뿐입니다. 선택 파일 `outputs/krreal/v1/sources.csv`(열: `source_id,kind,public,note`)에서 `public`을 `yes`로 적은 출처만 manifest에 `public_candidate`로 표시됩니다. 나머지는 `private`입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `shooting-guide.md` | 촬영 가이드: 층과 장수, 층별 촬영 방법, 피할 것, 파일 이름, 노트북으로 옮기기 |
| `kit.json` | 층, 휴대폰별 촬영 목록, 플래그, 글자 영역·기준선 종류, 파일 이름 규칙. 다른 파일이 모두 이 값을 씁니다 |
| `label.schema.json` | 라벨 JSON Schema(`krreal-label-1`) |
| `labeler.html` | 브라우저에서 여는 라벨러. 파일 하나이고 외부 자원을 쓰지 않습니다 |
| `transcript-template.txt` | 전사 파일 틀과 전사 규칙 |
| `krreal.py` | 명령: `shotlist`, `ingest`, `draft`, `transcripts`, `check`, `manifest`, `sync-labeler` |
| `kitlib.py`, `labelcheck.py`, `drafts.py` | 명령이 쓰는 모듈: 사진과 manifest, 라벨·전사 검사, 초안과 훅 |

`kit.json`을 고친 뒤에는 `krreal.py sync-labeler`로 라벨러 안의 사본을 맞춥니다. `tests/test_krreal_kit.py`가 스키마, 라벨러, `kit.json`이 서로 맞는지 확인합니다.

## 작업 순서

명령은 저장소 루트에서 `.venv/bin/python tools/krreal/krreal.py <명령>`으로 부릅니다. `--root`의 기본값은 `outputs/krreal/v1`입니다.

| 단계 | 명령 또는 도구 | 결과 |
|---|---|---|
| 1. 촬영 목록 | `shotlist` | `shotlist.csv` (휴대폰마다 30줄) |
| 2. 촬영 | `shooting-guide.md` | 휴대폰 사진 60장 |
| 3. 옮기기 | 가이드 7절, 그다음 `ingest` | `photos/kr1-….jpg` (원본 바이트 그대로) |
| 4. 목록 | `manifest` | `manifest.json`, 부족한 장수와 경고 |
| 5. 초안 | `draft [--provider …]` | `drafts/<이름>.json` |
| 6. 라벨 | `labeler.html` | `labels/<이름>.json` |
| 7. 검사 | `check [--drafts]` | 오류 목록, 종료 코드 |
| 8. 전사 | `transcripts [--ocr …]`, 텍스트 편집기 | `transcripts/<이름>.txt` |
| 9. 마감 | `manifest --strict` | `pilot complete: yes` |

종료 코드: 0은 완료, 1은 문제가 있음(표준 오류에 목록), 2는 사용법 오류입니다.

**파일럿 완료 기준**(`manifest`의 `summary.m1_missing`이 비면 완료)

- 층마다 계획한 장수 이상, 모두 60장 이상
- EXIF 기종이 다른 휴대폰 2대
- 모든 사진에 `labeled` 또는 `checked` 라벨이 있고 검사 오류가 없음
- 두 사람이 확인한(`checked`) 전사 20쪽 이상

## 라벨

라벨은 사진 한 장에 JSON 파일 하나이고 `label.schema.json`을 따릅니다. 스키마로 적을 수 없는 규칙은 `check`가 봅니다.

**좌표.** EXIF 방향을 적용한 사진의 픽셀입니다. (0, 0)은 왼쪽 위 픽셀의 중심이고 x는 오른쪽, y는 아래로 커집니다(`docs/INTERFACE_SPEC.md`와 같음). flattener 정규화 좌표는 `g = 2 * (p + 0.5) / size - 1`입니다. 페이지가 사진 밖으로 나간 경우 모서리를 사진 밖에 찍을 수 있습니다(사진 크기의 50%까지).

| 항목 | 규칙 |
|---|---|
| `quad` | 페이지 모서리 4점. 페이지 기준 읽기 방향으로 TL, TR, BR, BL. 화면에서 시계 방향이고 볼록해야 합니다 |
| `page_rotation_deg`, `page_quarter` | quad에서 계산합니다. 위쪽 변(TL→TR)과 아래쪽 변(BL→BR) 방향의 합이 이미지 x축과 이루는 시계 방향 각도, 그 각도에 가장 가까운 1/4 회전 수(0–3). 사진을 반시계 방향으로 `90 × page_quarter`도 돌리면 페이지가 똑바로 섭니다 |
| `curved_edges` | 휜 가장자리만 적습니다. 모서리 포함 7점 이상, 방향은 top: TL→TR, bottom: BL→BR, left: TL→BL, right: TR→BR. 끝점은 quad 모서리와 같습니다 |
| `text_polygons` | 글자 영역 다각형. 머리말과 쪽번호까지 감쌉니다. 종류 `text`, `table`, `formula` |
| `reference_lines` | 평평한 페이지에서 곧은 선 3–5개: `baseline`(글줄 기준선), `table_rule`(표선), `vertical_line`(세로선) |
| `page_count`, `pages` | 페이지 없음은 0, 두 쪽 펼침은 2. 두 쪽 펼침은 쪽마다 quad를 달고 `reading_order` 1이 왼쪽 쪽입니다 |
| `flags` | `occluded`, `cut_off`, `neighbor_page`, `vertical_text`, `no_page`, `two_page`, `exif_wrong`, `glare`, `shadow`, `blur`. `no_page`, `two_page`, `cut_off`는 쪽 수와 `cut_off_sides`에서 정해집니다 |
| `cut_off_sides` | 페이지가 사진 밖으로 나간 쪽(사진 기준 `top`, `right`, `bottom`, `left`) |
| `photo`, `capture` | 파일 이름, SHA-256, 크기, EXIF 방향, 기기와 촬영 값. GPS는 옮겨 적지 않습니다 |
| `review` | `draft` → `labeled`(작성자, 시각) → `checked`(다른 사람, 시각). 걸린 시간 `seconds_spent` |
| `draft` | 초안을 만든 provider와 채운 항목. 사람이 고친 뒤에도 남깁니다 |

`labeled`와 `checked`는 모든 규칙을 통과해야 합니다. 두 쪽 펼침(`spread`)은 이번 마일스톤에 quad만 있어도 됩니다. `draft`는 스키마와 이미 그린 도형의 기하만 봅니다.

## 라벨러 (`labeler.html`)

파일을 브라우저로 엽니다(Chrome, Edge, Firefox). 사진은 브라우저 안에서만 읽고 어디에도 보내지 않습니다.

1. **폴더 열기**로 `outputs/krreal/v1`을 고릅니다. `photos/`, `drafts/`, `labels/`, `manifest.json`을 함께 읽습니다. 사진과 JSON 파일을 **파일 열기**로 골라도 됩니다.
2. 위쪽 **이름** 칸에 이름을 적습니다.
3. 사진마다 모서리 4점(Q), 휜 가장자리(E), 글자 영역(T), 기준선 3–5개(L), 플래그를 답니다. 오른쪽 **검사**가 비면 저장합니다(Ctrl+S). 오류가 남아 있으면 초안(`draft`)으로 저장합니다.
4. 저장 위치
   - Chrome, Edge: **라벨 폴더 연결**로 `labels/` 폴더를 고르면 그 폴더에 바로 씁니다.
   - 그 밖: 내려받기 폴더에 `<이름>.json`이 생깁니다. `labels/`로 옮깁니다.
5. 다른 사람이 확인한 뒤 자기 이름으로 **검수 완료로 저장**을 누릅니다.

- 저장하지 않은 작업은 브라우저 저장소에 자동으로 남고, 그 사진을 다시 열면 불러옵니다.
- 사진을 열 때 SHA-256을 계산해 초안, 라벨, manifest의 값과 비교합니다. EXIF 방향을 적용한 크기가 기록과 다르면 경고합니다.
- 노란 굵은 선과 화살표가 페이지 위쪽입니다. 모서리를 다른 순서로 찍었으면 **모서리 순서 돌리기**(O)나 **좌우 바꾸기**(M)로 고칩니다.
- 단축키는 **도움말**(?)에 있습니다. 한 장에 1.5–2분이 목표입니다.

## 초안과 훅

`draft`는 사진마다 `drafts/<이름>.json`을 씁니다. 기본 provider `none`은 사진 정보(SHA-256, 크기, EXIF, 기기)만 채웁니다. `labels/`는 건드리지 않습니다.

`--provider module:function` 또는 `--provider path/to/file.py:function`으로 기하를 미리 채우는 함수를 붙입니다.

```python
def propose(photo: Path, info: dict) -> dict:
    # info: kitlib.photo_info(photo)와 name(PhotoName)
    # 모서리는 TL, TR, BR, BL. quad 말고는 모두 선택 항목입니다.
    page = {
        "quad": [[x, y], ...],
        "curved_edges": {"top": [...]},
        "text_polygons": [...],
        "reference_lines": [...],
    }
    return {"pages": [page], "flags": [...]}
```

좌표 규약은 라벨과 같습니다. 곡선 끝점은 quad 모서리로 바뀝니다. 검사를 통과하지 못한 제안은 버리고 사진 정보만 남깁니다. 스캐너 provider는 `--provider tools/krreal/scan_provider.py:propose`입니다. 스캐너의 최종 격자를 `drafts.pages_from_grid`로 quad와 휜 가장자리로 바꿉니다. 같은 사진의 스캔 기록이 `scan-quarter/`나 `scan-default/`에 있으면 그 격자를 쓰고, 없으면 스캐너를 돌립니다. 스캐너가 원본을 돌려준 사진은 기하 없이 둡니다. 글자 영역과 기준선은 아직 채우지 않습니다.

## 전사

```bash
.venv/bin/python tools/krreal/krreal.py transcripts                 # 층별로 고른 24장
.venv/bin/python tools/krreal/krreal.py transcripts --ocr tesseract # Tesseract kor+eng 초안
.venv/bin/python tools/krreal/krreal.py transcripts --photos kr1-p1-curved-b03-007.jpg
```

- 층마다 `kit.json`의 `transcribe` 장수만큼 고정 순서로 고릅니다(곡면 8, 작은 글씨 4, 표·수식 4, 회전 3, 평평함 3, 배경 2).
- 파일은 머리(`# key: value`), `---` 줄, 본문 순서입니다. 규칙은 틀 안에 있습니다: 사진 속 글줄 하나를 한 줄에, 읽기 순서로, 인쇄된 그대로 적고, 표는 칸 사이에 탭, 수식은 `[[수식]]`, 그림은 `[[그림]]`, 읽을 수 없는 글자는 `[[?]]`입니다.
- `draft` → `transcribed`(1차 전사자) → `checked`(다른 확인자). 사진을 100%로 확대해 대조합니다.
- `--ocr tesseract`는 Tesseract의 `kor` 언어 데이터가 있어야 합니다. `--ocr module:function`은 `(photo, info) -> list[str]` 함수를 받습니다.
- `--overwrite`는 아직 `draft`인 파일만 다시 씁니다.

## manifest

`manifest.json`(`krreal-manifest-1`)은 사진마다 파일 이름, SHA-256, 바이트 수, 휴대폰, 층, 촬영 목록의 변형, 출처, 분할, 공개 여부, 크기, EXIF 방향, 기기, 촬영 시각, 라벨과 전사의 SHA-256과 상태를 적습니다. 분할은 출처 단위이고 dev가 사진의 40%에 가깝게 고정 순서로 정합니다. 한번 정한 출처의 분할은 다시 만들어도 그대로입니다. `problems`에는 이름 규칙 위반, 같은 바이트의 사진, 번호 중복, 라벨·전사 오류가, `warnings`에는 EXIF 없음, GPS 있음, JPEG이 아닌 파일, 한 출처가 15%를 넘는 경우가 들어갑니다.
