# 출처와 이용 조건

확인일: 2026-09-26.

## 코드와 합성 자료

프로젝트 코드, 자체 문서 텍스트·배경·좌표·가중치: [Apache-2.0](LICENSE).
번들 패키지 라이선스: `Apache-2.0 AND OFL-1.1`.

글꼴: `NotoSansKR[wght].ttf`, Noto Sans KR 2.004, 가변 굵기 100–900. 2026-09-21 취득.
출처: [Google Fonts 고정 버전](https://github.com/google/fonts/tree/b38c5c93af322c45f633e17ac440ec1e6c94d489/ofl/notosanskr), [noto-cjk 원본](https://github.com/notofonts/noto-cjk/tree/523d033d6cb47f4a80c58a35753646f5c3608a78).
SIL OFL-1.1, © 2014–2021 Adobe, Reserved Font Name `Source`. 원문: [OFL.txt](src/flattener/assets/fonts/OFL.txt).
SHA-256: `194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252`.

## DocUNet

[저자 페이지](https://www3.cs.stonybrook.edu/~cvl/docunet.html)에서 사진 130장과 스캔 65장을 받았습니다.
사용 범위는 로컬 연구 평가입니다. 공개·상업 재배포에는 권리자 확인이 필요합니다.
취득일·URL·파일 목록·다운로드 해시: `outputs/dataset/manifest.json`의 `collection.sources.docunet`.
전송 경로는 HTTP이며 해시는 취득한 파일의 동일성 확인에 사용합니다.

표준 벤치마크 입력인 크롭 사진 130장은 [crop.zip](http://vision.cs.stonybrook.edu/~kema/docwarp/crop.zip)(SHA-256 `30e385f4a8c7bb47ec93cb61c73e40d91ed390fbf86b22cf1c5e4e8898df4e81`)에서 받아 `outputs/benchmarks/docunet/crop/`에 풀었습니다. 로컬 연구 평가 전용이며 재배포하지 않습니다. 파일별 해시와 스캔 경로: `outputs/benchmarks/docunet/manifest.json`.

인용: Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, Dimitris Samaras.
*DocUNet: Document Image Unwarping via A Stacked U-Net*. CVPR 2018.

## 표준 평가 코드 (OARDoc, SIFT flow)

`src/flattener/benchmark/matlab.py`와 `siftflow.py`는 [OARDoc](https://github.com/GreatV/oardoc/tree/9c45c0e783748554a83abea482924a2443294bef)의 `oardoc/metrics/uvdoc.py`·`oardoc/metrics/siftflow.py`를 수정한 코드입니다. [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © OARDoc authors. NumPy 재작성, 회색 변환 순서 교정, Windows(MSVC) 빌드 등 변경 사항은 각 파일 머리말에 적었습니다.

LD·AD에 쓰는 SIFT flow C++(Ce Liu, "All rights reserved")는 포함하지 않습니다. 사용자가 받은 DocUNet `eval.zip` 또는 UVDoc `eval_code`의 `SIFTflow` 소스에서 실행 시 빌드합니다. 단, `siftflow.py`의 coarse-to-fine 구동부(`SiftFlow._coarse_to_fine`, `_level_offset`)는 OARDoc과 마찬가지로 같은 패키지의 `SIFTflowc2f.m` 제어 흐름과 매개변수 일정(레벨별 창 크기·반복 수·gamma, 흐름 업샘플 식)을 Python으로 다시 구현한 것입니다. 원본 파일은 포함하지 않으며, 이 재구현을 배포에 포함할지는 소유자가 판단할 사항입니다.

`ocr.py`의 `prepare_like_pytesseract`는 [pytesseract](https://github.com/madmaze/pytesseract) 0.3.8의 `prepare()`([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © pytesseract authors)를 문장 단위로 따릅니다. OCR 입력을 공개 규약과 바이트 단위로 맞추기 위해서입니다. DocTr·UVDoc OCR 규약은 코드 복사 없이 재구현했고, OCR 문서 목록만 인용합니다. 보정에 쓴 PaperEdge·UVDoc 공개 결과 이미지는 로컬 평가에만 사용하며 저장소에 포함하지 않습니다.

## SyntheticDoc

Daniel Woortmann, Tanguy Magne, Olga Sorkine-Hornung, 2026. [공식 배포·라이선스](https://doi.org/10.3929/ethz-c-000801994): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 이미지·좌표·메타데이터 일부를 원본 그대로 보관합니다. 재배포 시 저자·출처·라이선스·변경 사항을 표시합니다.

`direction.zip`에는 좌표 규약 검사용 `train/0000000/uv_inverse.exr` 원본 1개와 취득·분석 기록을 보관합니다. UV 바이트는 변경하지 않았으며, 별도 검사 스크립트를 추가했습니다.

비교 진단의 [공식 모델 코드](https://github.com/tanguymagne/SyntheticDoc/tree/aabd9cf05f479781df432e3bd7e940f3e6716dbd)는 MIT, © 2026 Tanguy MAGNE입니다. 실험 ZIP에 코드와 LICENSE를 함께 보존합니다. 공식 `original_31x45.ckpt`는 로컬 연구 진단에만 사용하며, 별도 가중치 이용 조건이 확인될 때까지 재배포에서 제외합니다.

## UVDoc (PaddlePaddle)

[공식 모델 카드](https://huggingface.co/PaddlePaddle/UVDoc/tree/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3)는 Apache-2.0을 명시합니다. 비교용 배포 버전은 `16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3`이며, 원본 가중치는 변경하지 않습니다. 좌표 추출용 그래프에는 출력 1개를 추가하고 원본과의 수치 일치를 검사합니다. 가중치는 로컬·원격 실험 캐시에 보관합니다.

## SmartDoc 2015

[Challenge 1 저자 배포 v2.0.0](https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0): CC BY 4.0. 원문 이미지의 권리는 각 저자에게 있습니다. 저자·원문 출처·라이선스·배포 체크섬을 통합 manifest와 `files/smartdoc-source-*`에 보관합니다.

보정된 촬영 원본 30종과 background01–04의 사진 360장을 사용합니다. 원문별 분할을 고정한 후 리사이즈·조명·휨 증강을 적용했습니다. 네 모서리 주석은 평면 근사 정답으로 사용합니다. 촬영 템플릿에도 원래의 보정 오차가 남을 수 있습니다.

인용: Jean-Christophe Burie 외. *ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)*. ICDAR 2015. 전체 저자와 변경 이력: `collection.sources.smartdoc`.

자료는 `outputs/dataset`에 보관하며 패키지에 포함하지 않습니다. 통합 manifest의 `collection.sources`에 출처별 취득 범위·이용 조건·원본 기록과 현재 파일 경로를 보존합니다.

## 의존성

| 패키지 | 검증 버전 | wheel 라이선스 표현 |
| --- | --- | --- |
| [NumPy](https://pypi.org/project/numpy/2.5.3/) | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [Pillow](https://pypi.org/project/Pillow/12.3.0/) | 12.3.0 | MIT-CMU |
| [PyTorch](https://download.pytorch.org/whl/cpu/torch/) | 2.14.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |

pytest·Ruff·build·Hatchling: MIT. 버전·배포 URL·해시는 `uv.lock`에 고정합니다.
의존성의 원문 라이선스와 번들 고지는 설치된 `*.dist-info/licenses/`에서 확인합니다.
재배포 시 해당 고지를 유지하세요.
