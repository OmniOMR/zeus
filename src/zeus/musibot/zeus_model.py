"""Zeus behind the worker IPC contract.

This is the half that knows about music: it answers what the loaded snapshot
is, and turns a batch of executions into transcriptions written into page
folders. The transport is in `protocol.py`.
"""

from pathlib import Path
from typing import Protocol

from ..model.inference_options import InferenceOptions
from ..model.model_options import ModelOptions
from ..musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml
from .protocol import Execution, Reporter


class Predictor(Protocol):
    """What this needs of a model, which `Zeus` satisfies.

    Named as a protocol rather than importing `Zeus` so that nothing in this
    package imports TensorFlow: the loading happens in the command, and what
    arrives here is something that predicts.
    """

    model_options: ModelOptions

    def predict(
        self,
        images: list[bytes],
        inference_options: InferenceOptions,
        with_progress_bar: bool = False,
    ) -> list[str]: ...


INPUT_FILE = "image.jpg"
"""The file a model is given, relative to the subdivision folder it sits in."""

MUSICXML_FILE = "transcription.musicxml"
LMX_FILE = "transcription.lmx"


class InvalidExecution(Exception):
    """The execution asks for something this model cannot do."""


class ZeusMusibotModel:
    """One loaded snapshot, serving executions."""

    def __init__(
        self,
        zeus: Predictor,
        snapshot_path: Path,
        pages_dir: Path,
        inference_options: InferenceOptions,
        write_lmx: bool = True,
    ) -> None:
        """
        :param zeus: An already-loaded model. Loading happens before `ready` is
            sent, which is what makes readiness gate usefully: a worker head
            announces nothing and consumes no work until its model has spoken,
            so the seconds spent loading weights are seconds during which this
            worker is simply not offered any.
        """
        self.zeus = zeus
        self.snapshot_path = snapshot_path
        self.pages_dir = pages_dir
        self.inference_options = inference_options
        self.write_lmx = write_lmx
        self.model_options: ModelOptions = zeus.model_options

    def describe(self) -> dict:
        """The `model` object of the `ready` message.

        The *Model* is the source of truth for its own name, version and
        signature — it is the thing that knows them — and the *Worker Head*
        republishes this when it announces itself.
        """
        name, version = self.model_options.musibot_identity(self.snapshot_path)

        outputs = [MUSICXML_FILE]
        if self.write_lmx:
            outputs.append(LMX_FILE)

        return {
            "name": name,
            "version": version,
            "signature": {
                # Only the first subdivision instance can be named, because a
                # Signature is a flat list of paths and cannot say "every
                # staff, however many there are". See docs/musibot-model.md.
                "input": [f"{subdivision}/1/{INPUT_FILE}" for subdivision in self._subdivisions()],
                "output": [
                    f"{subdivision}/1/{output}"
                    for subdivision in self._subdivisions()
                    for output in outputs
                ],
            },
            # Zeus is a batched model: filling one forward pass with several
            # staves is most of what makes it fast.
            "supports_batching": True,
        }

    def _subdivisions(self) -> list[str]:
        return self.model_options.input_subdivisions

    def handle(self, executions: list[Execution], reporter: Reporter) -> None:
        """Transcribe a batch, reporting each execution separately.

        Executions in a batch may come from different pipelines and different
        pages. One that fails does not fail its batch-mates: reading and
        writing are per-execution, and only the forward pass is shared.
        """
        # Resolve everything first, so that a bad path fails its own execution
        # rather than the batch's single prediction call.
        loaded: list[tuple[Execution, Path, bytes]] = []
        for execution in executions:
            try:
                image_path = self._resolve_input(execution)
                loaded.append((execution, image_path, image_path.read_bytes()))
            except Exception as exception:  # noqa: BLE001 — reported, not raised
                reporter.failed(execution.execution_id, str(exception))

        if not loaded:
            return

        for execution, _, _ in loaded:
            reporter.progress(execution.execution_id, "transcribing")

        print(f"transcribing {len(loaded)} image(s)")
        predictions = self.zeus.predict(
            images=[image for _, _, image in loaded],
            inference_options=self.inference_options,
        )

        for (execution, image_path, _), lmx in zip(loaded, predictions, strict=True):
            try:
                self._write_outputs(image_path.parent, lmx)
            except LmxDecodingError as error:
                # A prediction is not required to be well-formed LMX. One
                # unreadable staff is an ordinary outcome and fails only itself.
                reporter.failed(execution.execution_id, str(error))
            except Exception as exception:  # noqa: BLE001 — reported, not raised
                reporter.failed(execution.execution_id, str(exception))
            else:
                reporter.completed(execution.execution_id)

    def _resolve_input(self, execution: Execution) -> Path:
        """Locate the one image this execution names, refusing anything else."""
        if len(execution.input) != 1:
            raise InvalidExecution(
                f"Zeus reads exactly one image per execution, "
                f"but {len(execution.input)} input file(s) were given."
            )

        relative = execution.input[0]

        # The model must confine itself to the page folders named in the
        # command it is executing. The Worker Head rejects escaping paths too,
        # but a Model that only ever ran behind one is a Model that has never
        # been checked.
        page_dir = (self.pages_dir / execution.page).resolve()
        image_path = (page_dir / relative).resolve()
        if not image_path.is_relative_to(page_dir):
            raise InvalidExecution(f"The input path {relative!r} escapes its page folder.")

        subdivision = self._subdivision_of(relative)
        if subdivision is not None and not self.model_options.accepts(subdivision):
            raise InvalidExecution(
                f"This model reads {', '.join(self._subdivisions())}, but was given {relative!r}."
            )

        if not image_path.is_file():
            raise InvalidExecution(f"There is no file at {relative!r}.")

        return image_path

    @staticmethod
    def _subdivision_of(relative_path: str) -> str | None:
        """`Staves/1/image.jpg` names the `Staves` subdivision.

        A path with no subdivision at all — the page-level `image.jpg` — yields
        None and is not refused here: whether a page is a reasonable thing to
        hand this model is a question the signature cannot express and the
        caller has already answered.
        """
        parts = Path(relative_path).parts
        return parts[0] if len(parts) > 1 else None

    def _write_outputs(self, destination: Path, lmx: str) -> None:
        """Write the transcription beside the image it came from.

        `Staves/1/image.jpg` yields `Staves/1/transcription.musicxml`, which is
        where the Musicorpus Specification puts a staff's transcription.
        """
        if self.write_lmx:
            # Written before the conversion, so that a prediction the decoder
            # rejects can still be inspected — which is when one most wants it.
            (destination / LMX_FILE).write_text(lmx + "\n", encoding="utf-8")

        musicxml = lmx_to_musicxml(lmx)
        (destination / MUSICXML_FILE).write_text(musicxml, encoding="utf-8")
