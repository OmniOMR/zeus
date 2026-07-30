"""The metrics Zeus can report, and how to name them.

`SER` is the metric: it is what training reports, what evaluation writes, and
what the prediction visualization sorts by unless told otherwise. The others
exist because a single number over every token answers only one question, and
occasionally you want a different one — but they are opt-in, chosen per run
rather than computed always.

Adding a metric means writing it here and naming it in `ALL_METRICS`. Nothing
else has to change: the CLI resolves whatever names this module publishes.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .symbol_error_rate import is_tuplet_token, symbol_error_rate

Computation = Callable[[list[str], list[str]], float]
"""Takes gold and predicted LMX strings and returns one number."""


@dataclass(frozen=True)
class Metric:
    """One named, computable number over a set of predictions."""

    name: str
    """As it appears on the command line, in `metrics.yaml`, and in
    tensorboard. Not a python identifier — `SERnotuplets` is the name."""

    description: str
    """One line, shown when the command line lists what is available."""

    compute: Computation
    """Gold and predictions in, one number out."""


def _symbol_error_rate_over(keep: Callable[[str], bool] | None) -> Computation:
    def compute(gold: list[str], pred: list[str]) -> float:
        return symbol_error_rate(gold, pred, keep=keep)

    return compute


def _is_pitch_token(token: str) -> bool:
    from lmx.tokenization.vocabulary import PITCH_TOKENS

    return token in frozenset(PITCH_TOKENS)


SER = Metric(
    name="SER",
    description="Symbol error rate over all LMX tokens. The default.",
    compute=_symbol_error_rate_over(None),
)

SER_NO_TUPLETS = Metric(
    name="SERnotuplets",
    description="Symbol error rate ignoring tuplet tokens.",
    compute=_symbol_error_rate_over(lambda token: not is_tuplet_token(token)),
)

SER_PITCH_ONLY = Metric(
    name="SERpitchonly",
    description="Symbol error rate over pitch tokens alone, ignoring rhythm and everything else.",
    compute=_symbol_error_rate_over(_is_pitch_token),
)


ALL_METRICS: dict[str, Metric] = {
    metric.name: metric for metric in (SER, SER_NO_TUPLETS, SER_PITCH_ONLY)
}
"""Every metric that can be asked for by name."""

DEFAULT_METRICS: list[Metric] = [SER]
"""What is computed when nothing was asked for.

Deliberately just the one. Every extra metric is another number to interpret
and another line in tensorboard, and a run that reports three numbers by
default teaches nobody which of them to watch.
"""


def resolve_metrics(names: Iterable[str]) -> list[Metric]:
    """Look metrics up by name, refusing anything unknown.

    Duplicates are dropped and the given order is kept, so what a user typed is
    the order they get.
    """
    resolved: list[Metric] = []

    for name in names:
        metric = ALL_METRICS.get(name)
        if metric is None:
            raise ValueError(
                f"Unknown metric {name!r}. Available metrics are: {', '.join(ALL_METRICS)}."
            )
        if metric not in resolved:
            resolved.append(metric)

    if not resolved:
        raise ValueError("No metrics were given.")

    return resolved


def parse_metric_names(text: str) -> list[Metric]:
    """Resolve a comma-separated list of metric names, as the CLI takes them."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise ValueError(f"No metric names in {text!r}.")
    return resolve_metrics(names)


def compute_metrics(
    metrics: Iterable[Metric],
    gold: list[str],
    pred: list[str],
) -> dict[str, float]:
    """Compute each metric, keyed by its name."""
    return {metric.name: metric.compute(gold, pred) for metric in metrics}
