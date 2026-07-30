"""Metrics computed over model predictions.

Zeus is LMX-first: every metric here compares token sequences, and none of
them decodes MusicXML. Evaluation at the MusicXML level — tree edit distance
and its relatives — is heavy, needs decoding, and belongs to a benchmarking
rig run after a model is trained rather than to the repository that trains it.

`SER` is the default and is what everything reports unless told otherwise:

    from zeus.evaluation import SER, SER_PITCH_ONLY

    ser = SER.compute(gold_lmx_strings, predicted_lmx_strings)
"""

from .metrics import (
    ALL_METRICS,
    DEFAULT_METRICS,
    SER,
    SER_NO_TUPLETS,
    SER_PITCH_ONLY,
    Metric,
    compute_metrics,
    parse_metric_names,
    resolve_metrics,
)
from .symbol_error_rate import symbol_error_rate

__all__ = [
    "ALL_METRICS",
    "DEFAULT_METRICS",
    "SER",
    "SER_NO_TUPLETS",
    "SER_PITCH_ONLY",
    "Metric",
    "compute_metrics",
    "parse_metric_names",
    "resolve_metrics",
    "symbol_error_rate",
]
