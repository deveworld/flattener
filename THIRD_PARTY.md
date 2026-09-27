# 출처와 이용 조건

확인일: 2026-09-26. 패키지 라이선스 표현은 `Apache-2.0 AND MIT AND OFL-1.1`입니다.

## 패키지에 포함되는 것

- **프로젝트 코드와 자체 합성 자료**(문서 텍스트·배경·좌표·가중치): [Apache-2.0](LICENSE).
- **글꼴** `NotoSansKR[wght].ttf`: Noto Sans KR 2.004, 가변 굵기 100–900, 2026-09-21 취득.
  출처: [Google Fonts 고정 버전](https://github.com/google/fonts/tree/b38c5c93af322c45f633e17ac440ec1e6c94d489/ofl/notosanskr), [noto-cjk 원본](https://github.com/notofonts/noto-cjk/tree/523d033d6cb47f4a80c58a35753646f5c3608a78).
  SIL OFL-1.1, © 2014–2021 Adobe, Reserved Font Name `Source`. 원문: [OFL.txt](src/flattener/assets/fonts/OFL.txt).
  SHA-256: `194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252`.
- **`benchmark/dir300.py`**: [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet)(비상업 조건)의 DIR300 규약(`ssim_ld_eval_DIR300.m`, `OCR_eval_DIR300.py`)은 직접 구현했고, 파일 짝짓기, OCR 목록(`ocr_img_DIR300.txt`의 90장), 상수만 사실로 인용합니다.
- **`benchmark/matlab.py`, `benchmark/siftflow.py`**: [OARDoc](https://github.com/GreatV/oardoc/tree/9c45c0e783748554a83abea482924a2443294bef)의 `oardoc/metrics/uvdoc.py`와 `oardoc/metrics/siftflow.py`를 고친 코드입니다. [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © OARDoc authors. 변경 사항은 각 파일 머리말에 있습니다.
  `siftflow.py`의 coarse-to-fine 구동부(`SiftFlow._coarse_to_fine`, `_level_offset`)는 OARDoc처럼 SIFT flow 패키지의 `SIFTflowc2f.m` 제어 흐름과 매개변수 일정을 Python으로 옮긴 것입니다. 이 부분의 배포 여부는 소유자가 정합니다.
- **`benchmark/ocr.py`의 `prepare_like_pytesseract`**: [pytesseract](https://github.com/madmaze/pytesseract) 0.3.8 `prepare()`를 문장 단위로 따릅니다. [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © pytesseract authors. OCR 입력을 공개 규약과 바이트 단위로 맞추기 위한 것입니다. DocTr·UVDoc OCR 규약은 직접 구현했고 OCR 문서 목록만 인용합니다.
- **`benchmark/uvdoc.py`**: [UVDoc](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82)의 `uvdocBenchmark_pred.py`(`bilinear_unwarping`과 저장 방식), `uvdocBenchmark_eval.py`, `eval/eval_code/evalScriptUVDoc.m`의 벤치마크 규약을 옮겼습니다. MIT, © 2023 Tanguy MAGNE. 고지 전문은 [LICENSES/UVDoc-MIT.txt](LICENSES/UVDoc-MIT.txt)에 있습니다.
- **`dewarp/model.py`**: [UVDoc](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82)의 `model.py`에서 파생했습니다. MIT, © 2023 Tanguy MAGNE. 고지 전문은 파일 머리말과 [LICENSES/UVDoc-MIT.txt](LICENSES/UVDoc-MIT.txt)에 있습니다.

## 사용자가 직접 받는 외부 코드와 가중치

아래 항목은 사용자가 받아 로컬에서 씁니다.

| 대상 | 조건 | 쓰는 곳 |
| --- | --- | --- |
| SIFT flow C++ (Ce Liu) | All rights reserved | LD·AD 계산. DocUNet `eval.zip` 또는 UVDoc `eval_code`의 `SIFTflow` 소스를 실행할 때 빌드 |
| UVDoc `model/best_model.pkl` | MIT. 학습 데이터(Doc3D, UVDoc)에 비상업 텍스처 포함 | `dewarp` 초기화와 비교. release 초기화는 공개본 SHA-256이 맞는 파일만 받고, 계보를 체크포인트에 기록 |
| [SyntheticDoc 공식 코드](https://github.com/tanguymagne/SyntheticDoc/tree/aabd9cf05f479781df432e3bd7e940f3e6716dbd) | MIT, © 2026 Tanguy MAGNE | 좌표 규약 확인 |
| SyntheticDoc `original_31x45.ckpt` | 가중치 조건은 따로 확인 필요 | 로컬 연구 진단 전용 |
| [PaddlePaddle UVDoc](https://huggingface.co/PaddlePaddle/UVDoc/tree/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3) | Apache-2.0 | 비교 실험 |
| PaperEdge·UVDoc 공개 결과 이미지 | 저자 배포물 | 평가기 보정 검사 |

UVDoc 가중치로 초기화한 release 가중치의 배포 여부는 위 학습 데이터 조건을 보고 소유자가 정합니다.

## 데이터

모두 로컬 연구·평가용으로 둡니다. 재배포가 필요하면 저자·출처·라이선스·변경 사항을 표시하고 권리자 조건을 따릅니다.

| 데이터 | 조건 | 용도 | 기록 |
| --- | --- | --- | --- |
| [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 사진 130장, 스캔 65장 | 저자 배포, 공개·상업 재배포는 권리자 확인 | 평가 | `outputs/benchmarks/docunet/manifest.json` |
| DocUNet [crop.zip](http://vision.cs.stonybrook.edu/~kema/docwarp/crop.zip) (SHA-256 `30e385f4a8c7bb47ec93cb61c73e40d91ed390fbf86b22cf1c5e4e8898df4e81`) | 위와 같음 | 표준 벤치마크 입력 | `outputs/benchmarks/docunet/manifest.json` |
| [SmartDoc 2015 Challenge 1 v2.0.0](https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0) | CC BY 4.0, 원문 이미지 권리는 각 저자 | 촬영 템플릿 30종, 사진 360장 | `outputs/benchmarks/smartdoc/manifest.json` |
| [SyntheticDoc](https://doi.org/10.3929/ethz-c-000801994) | CC BY 4.0, 하위 출처마다 조건이 다름 | release 학습(조건이 확인된 하위 출처만) | `tools/syntheticdoc/`가 쓰는 샘플별 라이선스 기록, 체크포인트 `.pt.json`의 `data_sources` |
| [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset) | 텍스처 CC BY-NC-ND 3.0, Laval HDR 비상업 | research 학습 전용 | data spec의 `license` |
| [UVDoc 데이터셋](https://github.com/tanguymagne/UVDoc-Dataset) | 비상업 텍스처(DeepFloyd IF), DTD 배경 연구용 | research 학습 전용 | data spec의 `license` |
| [UVDoc 벤치마크](https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip) 50샘플 (SHA-256 `8bf91a3e0d9bfa3655a8bcb38b74c1a8c02a9f1ba5ce34cfc51e5ae0660067ac`) | 저장소 MIT, 압축 파일에는 라이선스 파일 없음 | 평가 전용 | `outputs/benchmarks/uvdoc-bench/manifest.json` |
| [DIR300](https://github.com/fh2019ustc/DocGeoNet) | © Hao Feng, 비상업·동일조건변경허락 | 평가 전용 | – |

인용

- Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, Dimitris Samaras. *DocUNet: Document Image Unwarping via A Stacked U-Net*. CVPR 2018.
- Jean-Christophe Burie 외. *ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)*. ICDAR 2015. 전체 저자: `tools/locator_v0.py`의 `SMARTDOC_CITATION`.
- Daniel Woortmann, Tanguy Magne, Olga Sorkine-Hornung. *SyntheticDoc: A Large Synthetic Dataset for Document Unwarping and Illumination Correction*. ECCV 2026.
- Sagnik Das, Ke Ma, Zhixin Shu, Dimitris Samaras, Roy Shilkrot. *DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks*. ICCV 2019. (Doc3D)
- Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung. *UVDoc: Neural Grid-based Document Unwarping*. SIGGRAPH Asia 2023.
- Hao Feng, Wengang Zhou, Jiajun Deng, Yuechen Wang, Houqiang Li. *Geometric Representation Learning for Document Image Rectification*. ECCV 2022. (DIR300)

## 의존성

NumPy, Pillow, PyTorch와 개발 도구의 버전·해시는 `uv.lock`에, 라이선스 전문은 설치된 `*.dist-info/licenses/`에 있습니다. 재배포할 때 이 고지를 함께 둡니다.
