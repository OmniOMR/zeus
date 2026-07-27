import re
from collections.abc import Callable


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
    # a broken installation rather than an expected path. It was a bare
    # `except:` before, which would have swallowed a KeyboardInterrupt during
    # the import and silently dropped to the pure-python version — some hundred
    # times slower over a full evaluation, with nothing said about it.
    levenshtein_distance = levenshtein_distance_pure

tuplets_exceptions = {
    "tuplet:start",
    "tuplet:stop",
}
tuplets_exception_re = re.compile(r"^\d+in\d+$")


def ser_metric(gold: list[str], pred: list[str]) -> dict[str, float]:
    assert len(gold) == len(pred), "Gold and predicted data must have the same length"

    ser_errors, ser_total = 0, 0
    sert_errors, sert_total = 0, 0
    for gold_lmx, pred_lmx in zip(gold, pred, strict=True):
        gold_tokens = gold_lmx.rstrip("\r\n").split()
        pred_tokens = pred_lmx.rstrip("\r\n").split()

        ser_errors += levenshtein_distance(gold_tokens, pred_tokens)
        ser_total += len(gold_tokens)

        gold_tuplets = [
            x
            for x in gold_tokens
            if x not in tuplets_exceptions and not tuplets_exception_re.match(x)
        ]
        pred_tuplets = [
            x
            for x in pred_tokens
            if x not in tuplets_exceptions and not tuplets_exception_re.match(x)
        ]
        sert_errors += levenshtein_distance(gold_tuplets, pred_tuplets)
        sert_total += len(gold_tuplets)

    assert ser_total > 0, "Gold data cannot be empty"
    return {"SER": 100 * ser_errors / ser_total, "SERnotuplets": 100 * sert_errors / sert_total}
