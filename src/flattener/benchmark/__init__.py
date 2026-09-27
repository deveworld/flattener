"""Standard document-unwarping benchmarks with the calibrated MATLAB/OCR protocols.

``score_pair`` scores one rectified image against its scan (MS-SSIM always; LD and
AD with a SIFT-flow backend). ``evaluate_docunet`` and ``evaluate_ocr`` score a
directory of DocUNet predictions and return JSON-ready reports; ``predict_uvdoc`` and
``evaluate_uvdoc`` run the UVDoc benchmark protocol; ``predict_dir300``,
``evaluate_dir300`` and ``evaluate_dir300_ocr`` run DIR300. SIFT flow (Ce Liu)
is compiled at run time from sources the user supplies (``load_siftflow``). Command
line: ``python -m flattener.benchmark``.
"""

from .calibration import evaluate_gates, run_calibration, summarize_calibration
from .dir300 import evaluate_dir300, evaluate_dir300_ocr, predict_dir300
from .docunet import (
    CONVENTIONS,
    DEFAULT_WORKERS,
    REF_RESIZE_MODES,
    convention_views,
    evaluate_docunet,
    reference_resizes,
)
from .metrics import (
    AD_SOLVERS,
    DEFAULT_AD_SOLVER,
    METRIC_AREA,
    PairScore,
    PreparedReference,
    aligned_distortions,
    local_distortion,
    prepare_reference,
    read_image,
    score_pair,
    score_prepared,
)
from .ocr import (
    DEWARPNET_SETTING_2,
    DOCTR_SETTING_1,
    PROTOCOLS,
    Tesseract,
    character_error_rate,
    evaluate_ocr,
    levenshtein,
)
from .siftflow import SiftFlow, SiftFlowBuild, build_siftflow, load_siftflow
from .uvdoc import evaluate_uvdoc, predict_uvdoc

__all__ = [
    "AD_SOLVERS",
    "CONVENTIONS",
    "DEFAULT_AD_SOLVER",
    "DEFAULT_WORKERS",
    "DEWARPNET_SETTING_2",
    "DOCTR_SETTING_1",
    "METRIC_AREA",
    "PROTOCOLS",
    "REF_RESIZE_MODES",
    "PairScore",
    "PreparedReference",
    "SiftFlow",
    "SiftFlowBuild",
    "Tesseract",
    "aligned_distortions",
    "build_siftflow",
    "character_error_rate",
    "convention_views",
    "evaluate_dir300",
    "evaluate_dir300_ocr",
    "evaluate_docunet",
    "evaluate_gates",
    "evaluate_ocr",
    "evaluate_uvdoc",
    "levenshtein",
    "load_siftflow",
    "local_distortion",
    "predict_dir300",
    "predict_uvdoc",
    "prepare_reference",
    "read_image",
    "reference_resizes",
    "run_calibration",
    "score_pair",
    "score_prepared",
    "summarize_calibration",
]
