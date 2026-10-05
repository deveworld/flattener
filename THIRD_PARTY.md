# Sources and terms of use

Checked on 2026-09-26. The package license expression is `Apache-2.0 AND MIT AND OFL-1.1`.

## Included in the package

- **Project code and our own synthetic material** (document text, backgrounds, coordinates, weights): [Apache-2.0](LICENSE).
- **Font** `NotoSansKR[wght].ttf`: Noto Sans KR 2.004, variable weight 100–900, obtained on 2026-09-21.
  Sources: [Google Fonts, pinned version](https://github.com/google/fonts/tree/b38c5c93af322c45f633e17ac440ec1e6c94d489/ofl/notosanskr), [noto-cjk original](https://github.com/notofonts/noto-cjk/tree/523d033d6cb47f4a80c58a35753646f5c3608a78).
  SIL OFL-1.1, © 2014–2021 Adobe, Reserved Font Name `Source`. Full text: [OFL.txt](src/flattener/assets/fonts/OFL.txt).
  SHA-256: `194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252`.
- **`benchmark/dir300.py`**: the DIR300 protocol of [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet) (non-commercial terms), `ssim_ld_eval_DIR300.m` and `OCR_eval_DIR300.py`, is our own implementation. Only facts are quoted from it: the file pairing, the OCR list (90 images in `ocr_img_DIR300.txt`) and the constants.
- **`benchmark/matlab.py`, `benchmark/siftflow.py`**: modified from `oardoc/metrics/uvdoc.py` and `oardoc/metrics/siftflow.py` in [OARDoc](https://github.com/GreatV/oardoc/tree/9c45c0e783748554a83abea482924a2443294bef). [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © OARDoc authors. The changes are listed in the header of each file.
  The coarse-to-fine driver in `siftflow.py` (`SiftFlow._coarse_to_fine`, `_level_offset`) ports the control flow and parameter schedule of `SIFTflowc2f.m` from the SIFT flow package to Python, as OARDoc does. The owner decides whether this part is distributed.
- **`prepare_like_pytesseract` in `benchmark/ocr.py`**: follows `prepare()` of [pytesseract](https://github.com/madmaze/pytesseract) 0.3.8 statement by statement. [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), © pytesseract authors. It makes the OCR input byte-identical to the public protocol. The DocTr and UVDoc OCR protocols are our own implementations; only their OCR document lists are quoted.
- **`benchmark/uvdoc.py`**: ports the benchmark protocol of `uvdocBenchmark_pred.py` (`bilinear_unwarping` and the way results are saved), `uvdocBenchmark_eval.py` and `eval/eval_code/evalScriptUVDoc.m` from [UVDoc](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82). MIT, © 2023 Tanguy MAGNE. The full notice is in [LICENSES/UVDoc-MIT.txt](LICENSES/UVDoc-MIT.txt).
- **`dewarp/model.py`**: derived from `model.py` in [UVDoc](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82). MIT, © 2023 Tanguy MAGNE. The full notice is in the file header and in [LICENSES/UVDoc-MIT.txt](LICENSES/UVDoc-MIT.txt).

## External code and weights that users download themselves

Users download the items below and use them locally.

| Item | Terms | Used for |
| --- | --- | --- |
| SIFT flow C++ (Ce Liu) | All rights reserved | LD and AD computation. Built when running the `SIFTflow` source in DocUNet `eval.zip` or UVDoc `eval_code` |
| UVDoc `model/best_model.pkl` | MIT. The training data (Doc3D, UVDoc) include non-commercial textures | Initializing `dewarp` and comparisons. Release initialization accepts only a file whose SHA-256 matches the public release, and records the lineage in the checkpoint |
| [SyntheticDoc official code](https://github.com/tanguymagne/SyntheticDoc/tree/aabd9cf05f479781df432e3bd7e940f3e6716dbd) | MIT, © 2026 Tanguy MAGNE | Checking the coordinate convention |
| SyntheticDoc `original_31x45.ckpt` | Terms for the weights still need to be checked | Local research diagnostics only |
| [PaddlePaddle UVDoc](https://huggingface.co/PaddlePaddle/UVDoc/tree/16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3) | Apache-2.0 | Comparison experiments |
| Published PaperEdge and UVDoc result images | Distributed by the authors | Evaluator calibration checks |

The owner decides whether release weights initialized from the UVDoc weights are distributed, based on the training data terms above.

## Data

All of it is kept for local research and evaluation. If it has to be redistributed, credit the authors, source, license and changes, and follow the rights holders' terms.

| Data | Terms | Use | Record |
| --- | --- | --- | --- |
| [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 130 photos, 65 scans | Distributed by the authors; public or commercial redistribution needs confirmation from the rights holders | Evaluation | `outputs/benchmarks/docunet/manifest.json` |
| DocUNet [crop.zip](http://vision.cs.stonybrook.edu/~kema/docwarp/crop.zip) (SHA-256 `30e385f4a8c7bb47ec93cb61c73e40d91ed390fbf86b22cf1c5e4e8898df4e81`) | Same as above | Standard benchmark input | `outputs/benchmarks/docunet/manifest.json` |
| [SmartDoc 2015 Challenge 1 v2.0.0](https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/tag/v2.0.0) | CC BY 4.0; rights to the source document images stay with their authors | 30 capture templates, 360 photos | `outputs/benchmarks/smartdoc/manifest.json` |
| [SyntheticDoc](https://doi.org/10.3929/ethz-c-000801994) | CC BY 4.0; terms differ between sub-sources | Release training (only sub-sources whose terms have been confirmed) | Per-sample license records written by `tools/syntheticdoc/`, `data_sources` in the checkpoint's `.pt.json` |
| [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset) | Textures CC BY-NC-ND 3.0, Laval HDR non-commercial | Research training only | `license` in the data spec |
| [UVDoc dataset](https://github.com/tanguymagne/UVDoc-Dataset) | Non-commercial textures (DeepFloyd IF), DTD backgrounds for research use | Research training only | `license` in the data spec |
| [UVDoc benchmark](https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip) 50 samples (SHA-256 `8bf91a3e0d9bfa3655a8bcb38b74c1a8c02a9f1ba5ce34cfc51e5ae0660067ac`) | Repository MIT; the archive contains no license file | Evaluation only | `outputs/benchmarks/uvdoc-bench/manifest.json` |
| [DIR300](https://github.com/fh2019ustc/DocGeoNet) | © Hao Feng, non-commercial, share-alike | Evaluation only | – |
| [Poly Haven](https://polyhaven.com) surface textures | CC0 | Background synthesis for release training | `manifest.json` written by `tools/backgrounds/fetch_polyhaven.py` (asset URLs, authors, hashes) |

Citations

- Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, Dimitris Samaras. *DocUNet: Document Image Unwarping via A Stacked U-Net*. CVPR 2018.
- Jean-Christophe Burie et al. *ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)*. ICDAR 2015. Full author list: `SMARTDOC_CITATION` in `tools/locator_v0.py`.
- Daniel Woortmann, Tanguy Magne, Olga Sorkine-Hornung. *SyntheticDoc: A Large Synthetic Dataset for Document Unwarping and Illumination Correction*. ECCV 2026.
- Sagnik Das, Ke Ma, Zhixin Shu, Dimitris Samaras, Roy Shilkrot. *DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks*. ICCV 2019. (Doc3D)
- Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung. *UVDoc: Neural Grid-based Document Unwarping*. SIGGRAPH Asia 2023.
- Hao Feng, Wengang Zhou, Jiajun Deng, Yuechen Wang, Houqiang Li. *Geometric Representation Learning for Document Image Rectification*. ECCV 2022. (DIR300)

## Dependencies

Versions and hashes of NumPy, Pillow, PyTorch and the development tools are in `uv.lock`. Full license texts are in the installed `*.dist-info/licenses/`. Keep this notice with any redistribution.
