"""Tests for the early stopping regime of `zeus train`.

The decision of when to stop and which snapshots to delete is what decides
which model a run ends with, and getting it wrong is silent: a run that stops a
few epochs early, or deletes the snapshot it later copies, still finishes. So
the bookkeeping is pure and pinned down here, and the training loop only obeys.
"""

import argparse

import pytest

from zeus.cli.run import build_parser
from zeus.cli.train_command import resolve_lr_decay
from zeus.model.early_stopping import EarlyStopping, is_evaluation_tick, snapshots_to_collect
from zeus.model.training_options import TrainingOptions


def run_through(early_stopping: EarlyStopping, sers: list[float]) -> int:
    """Feeds 1-based epochs in until told to stop; returns the stopping epoch."""
    for epoch, ser in enumerate(sers, start=1):
        early_stopping.observe(epoch, ser)
        if early_stopping.should_stop(epoch):
            return epoch
    raise AssertionError("did not stop")


def test_it_continues_patience_epochs_past_the_minimum() -> None:
    early_stopping = EarlyStopping(patience=3)

    stopped_at = run_through(early_stopping, [0.5, 0.4, 0.3, 0.35, 0.32, 0.31, 0.6])

    assert early_stopping.best_epoch == 3
    assert stopped_at == 6


def test_a_new_minimum_within_the_patience_resets_it() -> None:
    early_stopping = EarlyStopping(patience=2)

    stopped_at = run_through(early_stopping, [0.5, 0.6, 0.4, 0.45, 0.5])

    assert early_stopping.best_epoch == 3
    assert stopped_at == 5


def test_a_plateau_is_not_an_improvement() -> None:
    """Equal is not lower, so the earliest of equal epochs is the best."""
    early_stopping = EarlyStopping(patience=2)

    stopped_at = run_through(early_stopping, [0.3, 0.3, 0.3])

    assert early_stopping.best_epoch == 1
    assert stopped_at == 3


def test_patience_must_be_positive() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_evaluation_ticks_follow_from_and_each() -> None:
    ticks = [epoch for epoch in range(1, 21) if is_evaluation_tick(epoch, 5, 5)]

    assert ticks == [5, 10, 15, 20]


def test_ticks_before_evaluation_from_do_not_count() -> None:
    ticks = [epoch for epoch in range(1, 21) if is_evaluation_tick(epoch, 12, 5)]

    assert ticks == [15, 20]


def test_only_snapshots_older_than_the_patience_are_collected() -> None:
    early_stopping = EarlyStopping(patience=2)
    for epoch in range(1, 7):
        early_stopping.observe(epoch, 1.0 / epoch)

    collected = snapshots_to_collect(range(1, 7), 6, early_stopping, 100, 100)

    assert collected == [1, 2, 3]


def test_evaluation_ticks_survive_collection() -> None:
    early_stopping = EarlyStopping(patience=1)
    for epoch in range(1, 11):
        early_stopping.observe(epoch, 1.0 / epoch)

    collected = snapshots_to_collect(range(1, 11), 10, early_stopping, 4, 2)

    assert collected == [1, 2, 3, 5, 7]


def test_the_best_epoch_survives_collection() -> None:
    """It is always within the patience while training runs, but it is the one
    snapshot the run ends by copying, so it is protected explicitly."""
    early_stopping = EarlyStopping(patience=2)
    early_stopping.observe(1, 0.1)

    collected = snapshots_to_collect([1, 2, 3, 4, 5], 5, early_stopping, 100, 100)

    assert 1 not in collected


def test_collection_keeps_every_epoch_the_best_could_still_be_in() -> None:
    """Simulates a whole run: at the stop, the best snapshot still exists."""
    sers = [0.9, 0.5, 0.7, 0.4, 0.6, 0.45, 0.5, 0.8]
    early_stopping = EarlyStopping(patience=3)
    stored: set[int] = set()
    for epoch, ser in enumerate(sers, start=1):
        stored.add(epoch)
        early_stopping.observe(epoch, ser)
        stored -= set(snapshots_to_collect(stored, epoch, early_stopping, 1, 3))
        if early_stopping.should_stop(epoch):
            break

    assert early_stopping.best_epoch == 4
    assert stored == {3, 4, 5, 6, 7}


def test_training_options_need_exactly_one_regime() -> None:
    with pytest.raises(ValueError):
        TrainingOptions(epochs=None, evaluation_from=1, evaluation_each=1, is_finetuning=True)
    with pytest.raises(ValueError):
        TrainingOptions(
            epochs=10,
            evaluation_from=1,
            evaluation_each=1,
            is_finetuning=True,
            lr_decay="none",
            early_stopping_patience=3,
        )


def test_training_options_refuse_decay_with_early_stopping() -> None:
    with pytest.raises(ValueError):
        TrainingOptions(
            epochs=None,
            evaluation_from=1,
            evaluation_each=1,
            is_finetuning=True,
            lr_decay="cos",
            early_stopping_patience=3,
        )


REQUIRED = ["train", "--experiment", "x", "--train", "t.pickle"]


def parse(*flags: str) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    return parser, parser.parse_args([*REQUIRED, *flags])


def test_fixed_epochs_decay_by_default() -> None:
    parser, args = parse("--epochs", "10")

    assert resolve_lr_decay(parser, args) == "cos"


def test_early_stopping_defaults_to_a_fixed_rate() -> None:
    parser, args = parse("--early-stopping", "s.pickle", "--stopping-patience", "5")

    assert resolve_lr_decay(parser, args) == "none"


@pytest.mark.parametrize(
    "flags",
    [
        ["--early-stopping", "s.pickle"],
        ["--early-stopping", "s.pickle", "--stopping-patience", "0"],
        ["--early-stopping", "s.pickle", "--stopping-patience", "5", "--lr-decay", "cos"],
        ["--epochs", "10", "--stopping-patience", "5"],
    ],
    ids=["no-patience", "zero-patience", "with-decay", "patience-without-stopping"],
)
def test_incoherent_regimes_are_refused(flags: list[str]) -> None:
    parser, args = parse(*flags)

    with pytest.raises(SystemExit):
        resolve_lr_decay(parser, args)


@pytest.mark.parametrize(
    "flags",
    [[], ["--epochs", "10", "--early-stopping", "s.pickle", "--stopping-patience", "5"]],
    ids=["neither", "both"],
)
def test_exactly_one_of_epochs_and_early_stopping_is_required(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(*flags)
