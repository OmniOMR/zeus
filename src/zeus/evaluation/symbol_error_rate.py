"""Symbol error rate over LMX token sequences.

Every metric Zeus computes is a symbol error rate over some subset of the
tokens, so this is the one calculation and the metrics differ only in which
tokens they keep. See `metrics.py` for the ones that have names.
"""

import re
from collections.abc import Callable

TokenFilter = Callable[[str], bool]
"""Decides whether one LMX token takes part in a metric."""


def levenshtein_distance_pure(a: list, b: list) -> int:
    len_a, len_b = len(a), len(b)

    distances = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        prev_distances, distances = distances, [i] + [0] * len_b
        for j in range(1, len_b + 1):
            distances[j] = min(
                distances[j - 1] + 1,  # insertion
                prev_distances[j] + 1,  # deletion
                prev_distances[j - 1] + (a[i - 1] != b[j - 1]),  # substitution or match
            )

    return distances[-1]


try:
    import Levenshtein

    levenshtein_distance: Callable[[list, list], int] = Levenshtein.distance
except ImportError:
    # The C implementation is a declared dependency, so this is a fallback for
    # a broken installation rather than an expected path.
    levenshtein_distance = levenshtein_distance_pure


TUPLET_TOKENS = frozenset({"tuplet:start", "tuplet:stop"})

TIME_MODIFICATION_PATTERN = re.compile(r"^\d+in\d+$")
"""Tuplet ratios such as `3in2`. Matched by shape rather than against the LMX
vocabulary's list of them, because this is what the 2024 experiments used and
the numbers have to stay comparable to what was published."""


def is_tuplet_token(token: str) -> bool:
    return token in TUPLET_TOKENS or TIME_MODIFICATION_PATTERN.match(token) is not None


def tokenize(lmx: str) -> list[str]:
    """Split one LMX string into its tokens."""
    return lmx.rstrip("\r\n").split()


def symbol_error_rate(
    gold: list[str],
    pred: list[str],
    keep: TokenFilter | None = None,
) -> float:
    """The percentage of gold tokens that the predictions got wrong.

    Edit distance between predicted and gold token sequences, summed over the
    whole corpus and divided once at the end. That is deliberate and is not the
    same as averaging per-sample rates: a long sample and a short one should
    not weigh the same, and a corpus-level rate is what the literature reports.

    Insertions count as errors while contributing nothing to the total, so a
    model that predicts far too much can score above 100.

    :param gold: Gold LMX strings, one per sample.
    :param pred: Predicted LMX strings, in the same order.
    :param keep: Restricts the comparison to the tokens it accepts. `None`
        compares everything.
    """
    if len(gold) != len(pred):
        raise ValueError(
            f"Gold and predicted data must have the same length, got {len(gold)} and {len(pred)}."
        )

    errors, total = 0, 0
    for gold_lmx, pred_lmx in zip(gold, pred, strict=True):
        gold_tokens = tokenize(gold_lmx)
        pred_tokens = tokenize(pred_lmx)

        if keep is not None:
            gold_tokens = [token for token in gold_tokens if keep(token)]
            pred_tokens = [token for token in pred_tokens if keep(token)]

        errors += levenshtein_distance(gold_tokens, pred_tokens)
        total += len(gold_tokens)

    if total == 0:
        # Summing before dividing means this is a statement about the whole
        # corpus, not about one awkward sample: there is nothing anywhere for
        # this metric to measure.
        raise ValueError(
            "There are no gold tokens to compare. Either the gold data is "
            "empty, or this metric keeps no tokens that appear in it."
        )

    return 100 * errors / total
