"""Tests for the evaluation metrics.

Pure functions of token strings, so no model, no TensorFlow, and no dataset.
"""

import pytest

from zeus.evaluation import (
    ALL_METRICS,
    DEFAULT_METRICS,
    SER,
    SER_NO_TUPLETS,
    SER_PITCH_ONLY,
    compute_metrics,
    parse_metric_names,
    resolve_metrics,
    symbol_error_rate,
)

# --- SER, which must not have moved --------------------------------------


def test_a_perfect_prediction_scores_zero() -> None:
    assert SER.compute(["measure C4 quarter"], ["measure C4 quarter"]) == 0.0


def test_one_wrong_token_in_three() -> None:
    assert SER.compute(["a b c"], ["a x c"]) == pytest.approx(100 / 3)


def test_the_rate_is_over_the_corpus_not_the_average_of_samples() -> None:
    """A long sample and a short one must not weigh the same.

    One error in a four-token sample and none in a one-token sample is one
    error in five tokens — 20% — not the average of 25% and 0%.
    """
    rate = SER.compute(["a b c d", "e"], ["a b c X", "e"])

    assert rate == pytest.approx(20.0)


def test_predicting_too_much_can_exceed_one_hundred() -> None:
    """Insertions are errors but add nothing to the denominator."""
    assert SER.compute(["a"], ["a b c d"]) == pytest.approx(300.0)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        SER.compute(["a", "b"], ["a"])


def test_nothing_to_measure_is_refused() -> None:
    with pytest.raises(ValueError, match="no gold tokens"):
        SER.compute([""], ["a b c"])


# --- SERnotuplets ---------------------------------------------------------


def test_tuplet_ratios_are_ignored() -> None:
    gold, pred = ["a 3in2 b"], ["a b"]

    assert SER.compute(gold, pred) == pytest.approx(100 / 3)
    assert SER_NO_TUPLETS.compute(gold, pred) == 0.0


def test_tuplet_brackets_are_ignored() -> None:
    gold, pred = ["a tuplet:start b tuplet:stop"], ["a b"]

    assert SER_NO_TUPLETS.compute(gold, pred) == 0.0


def test_a_real_error_still_counts_without_tuplets() -> None:
    assert SER_NO_TUPLETS.compute(["a 3in2 b"], ["a x"]) == pytest.approx(50.0)


# --- SERpitchonly ---------------------------------------------------------


def test_rhythm_errors_do_not_count_towards_pitch() -> None:
    gold = ["measure C4 quarter D4 half"]
    pred = ["measure C4 whole D4 eighth"]

    assert SER_PITCH_ONLY.compute(gold, pred) == 0.0
    assert SER.compute(gold, pred) == pytest.approx(40.0)


def test_a_wrong_pitch_counts() -> None:
    gold = ["measure C4 quarter D4 half"]

    assert SER_PITCH_ONLY.compute(gold, ["measure C4 quarter E4 half"]) == pytest.approx(50.0)


def test_a_missing_pitch_counts() -> None:
    assert SER_PITCH_ONLY.compute(["C4 D4"], ["C4"]) == pytest.approx(50.0)


def test_a_prediction_with_no_pitches_at_all_is_wholly_wrong() -> None:
    assert SER_PITCH_ONLY.compute(["C4 D4"], ["measure rest"]) == pytest.approx(100.0)


def test_data_without_pitches_has_nothing_to_measure() -> None:
    with pytest.raises(ValueError, match="no gold tokens"):
        SER_PITCH_ONLY.compute(["measure rest"], ["measure rest"])


# --- the registry ---------------------------------------------------------


def test_ser_is_the_default_and_the_only_one() -> None:
    """Every extra default is another number nobody asked for."""
    assert DEFAULT_METRICS == [SER]


def test_every_metric_is_registered_under_its_own_name() -> None:
    for name, metric in ALL_METRICS.items():
        assert metric.name == name


def test_names_are_resolved_in_the_order_given() -> None:
    resolved = resolve_metrics(["SERpitchonly", "SER"])

    assert [metric.name for metric in resolved] == ["SERpitchonly", "SER"]


def test_a_repeated_name_is_only_computed_once() -> None:
    assert resolve_metrics(["SER", "SER"]) == [SER]


def test_an_unknown_name_lists_what_is_available() -> None:
    with pytest.raises(ValueError, match="Unknown metric 'TEDn'.*SER, SERnotuplets"):
        resolve_metrics(["TEDn"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SER", ["SER"]),
        ("SER,SERpitchonly", ["SER", "SERpitchonly"]),
        (" SER , SERnotuplets ", ["SER", "SERnotuplets"]),
        ("SER,SER", ["SER"]),
    ],
)
def test_the_comma_separated_form_the_cli_takes(text: str, expected: list[str]) -> None:
    assert [metric.name for metric in parse_metric_names(text)] == expected


def test_an_empty_metrics_argument_is_refused() -> None:
    with pytest.raises(ValueError, match="No metric names"):
        parse_metric_names(" , ")


def test_several_metrics_are_returned_keyed_by_name() -> None:
    computed = compute_metrics([SER, SER_PITCH_ONLY], ["C4 quarter"], ["C4 whole"])

    assert computed == {"SER": pytest.approx(50.0), "SERpitchonly": 0.0}


# --- the shared calculation ------------------------------------------------


def test_the_filter_applies_to_both_sides() -> None:
    """Otherwise a token the metric ignores in gold would count as an
    insertion when the model predicts it."""
    kept = symbol_error_rate(["a b"], ["a IGNORED b"], keep=lambda token: token != "IGNORED")

    assert kept == 0.0
