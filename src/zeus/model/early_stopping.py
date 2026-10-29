"""The bookkeeping of early stopping, kept apart from the training loop.

Nothing here touches TensorFlow or the disk. `Zeus.train` measures SER on the
early stopping dataset after every epoch and asks this module two questions:
whether to stop, and which epoch snapshots it no longer needs. Keeping the
answers pure is what lets them be tested in milliseconds rather than by
training a model.
"""

from collections.abc import Iterable


class EarlyStopping:
    """Tracks the validation minimum and decides when training is over.

    An epoch counts as an improvement only when its SER is strictly lower than
    every epoch before it, so a plateau does not keep training alive. Training
    stops once `patience` epochs have passed since the latest minimum without a
    new one — the patience is how much jitter in the SER curve is tolerated
    before a rise is believed.
    """

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError(f"Early stopping patience must be at least 1, got {patience}.")
        self.patience = patience
        """How many epochs to continue past the latest minimum."""

        self.best_epoch: int | None = None
        """The 1-based epoch with the lowest SER seen so far."""

        self.best_ser: float | None = None
        """The SER of that epoch."""

        self.history: dict[int, float] = {}
        """Every observed epoch and its SER, in the order observed."""

    def observe(self, epoch: int, ser: float) -> bool:
        """Records the SER of a finished 1-based epoch.

        Returns True when this epoch is a new minimum.
        """
        self.history[epoch] = ser
        if self.best_ser is None or ser < self.best_ser:
            self.best_epoch = epoch
            self.best_ser = ser
            return True
        return False

    def should_stop(self, epoch: int) -> bool:
        """Whether the patience has run out at this finished epoch."""
        if self.best_epoch is None:
            return False
        return epoch - self.best_epoch >= self.patience


def is_evaluation_tick(epoch: int, evaluation_from: int, evaluation_each: int) -> bool:
    """Whether a 1-based epoch is one `--evaluation-from` and `--evaluation-each`
    select for evaluation on the dev datasets and for a kept snapshot."""
    return epoch >= evaluation_from and epoch % evaluation_each == 0


def snapshots_to_collect(
    stored_epochs: Iterable[int],
    epoch: int,
    early_stopping: EarlyStopping,
    evaluation_from: int,
    evaluation_each: int,
) -> list[int]:
    """Which stored epoch snapshots can be deleted after a finished epoch.

    Under early stopping every epoch is stored, because any of them may turn
    out to be the best. Once an epoch is further back than the patience it can
    no longer become the result, so it is deleted — unless it is an evaluation
    tick, which is kept exactly as fixed-epoch training would keep it, or it is
    the current best, which the run will end by copying.
    """
    oldest_candidate = epoch - early_stopping.patience
    return sorted(
        stored
        for stored in stored_epochs
        if stored < oldest_candidate
        and stored != early_stopping.best_epoch
        and not is_evaluation_tick(stored, evaluation_from, evaluation_each)
    )
