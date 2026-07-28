"""The worker IPC contract, as transport.

This module knows about JSON lines on two pipes and nothing about music. It is
the half of `zeus musibot` that could serve any model at all, kept apart from
the half that runs Zeus so that both can be read — and tested — on their own.

See the contract itself at
https://github.com/OmniOMR/musibot/blob/main/docs/worker-ipc.md
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

IPC_VERSION = 1
"""The protocol version this model speaks. A *Worker Head* checks it for exact
equality and refuses anything else rather than guessing, so it moves only when
the contract does."""


@dataclass
class Execution:
    """One unit of work: read these files of this page, write the outputs."""

    execution_id: str
    """Opaque to us, echoed back in the report. It is how the head knows which
    of the executions in flight has finished."""

    page: str
    """The *Musicorpus Page* id, which is a folder under the pages directory."""

    input: list[str] = field(default_factory=list)
    """Paths within the page folder, e.g. `Staves/1/image.jpg`."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Free-form, model-specific, passed through from the *Pipeline*."""

    @staticmethod
    def from_message(message: dict[str, Any]) -> "Execution":
        return Execution(
            execution_id=str(message["execution_id"]),
            page=str(message["page"]),
            input=list(message.get("input") or []),
            parameters=dict(message.get("parameters") or {}),
        )


class Reporter:
    """How a handler reports on the executions it was given.

    Every execution must be reported exactly once, or the *Worker Head* waits
    for an answer that never comes. `serve` enforces that from the outside, so
    a handler that returns early or raises does not hang the head.
    """

    def __init__(self, results: TextIO) -> None:
        self._results = results
        self._reported: set[str] = set()

    @property
    def reported(self) -> set[str]:
        return set(self._reported)

    def completed(self, execution_id: str) -> None:
        self._send_report({"type": "completed", "execution_id": execution_id}, execution_id)

    def failed(self, execution_id: str, error: str) -> None:
        """The error string reaches the *Pipeline Execution* log, so it is
        worth writing for a human."""
        self._send_report(
            {"type": "failed", "execution_id": execution_id, "error": error}, execution_id
        )

    def progress(self, execution_id: str, message: str, fraction: float | None = None) -> None:
        """Attribute progress to one execution.

        Anything printed is already captured as a log line, so this is only
        needed inside a batch, where the head cannot tell which sample a
        printed line belongs to and attributes it to all of them.
        """
        report: dict[str, Any] = {
            "type": "progress",
            "execution_id": execution_id,
            "message": message,
        }
        if fraction is not None:
            report["fraction"] = fraction
        send(self._results, report)

    def _send_report(self, message: dict[str, Any], execution_id: str) -> None:
        if execution_id in self._reported:
            # A second report for one execution would be ignored by the head
            # with a warning; not sending it keeps the noise out of the log.
            return
        self._reported.add(execution_id)
        send(self._results, message)


Handler = Callable[[list[Execution], Reporter], None]
"""What `serve` calls for each command: do the work, report each execution."""


def send(results: TextIO, message: dict[str, Any]) -> None:
    """Put one message on the result pipe.

    The flush is not optional. A pipe is block-buffered, so an unflushed
    message is not late but invisible, and the *Worker Head* waits forever —
    the single easiest way to get a *Model* wrong.
    """
    results.write(json.dumps(message) + "\n")
    results.flush()


def serve(
    commands: TextIO,
    results: TextIO,
    description: dict[str, Any],
    handler: Handler,
) -> None:
    """Announce this model, then serve commands until told to stop.

    One command at a time, in a plain loop — a *Model* does not multitask, and
    the *Worker Head* sends nothing further until this one is reported. So
    there is no concurrency here of any kind.

    :param description: The `model` object of the `ready` message: name,
        version, signature and whether batching is supported.
    :param handler: Called with the executions of one command. It must report
        each of them, and anything it leaves unreported is failed on its behalf.
    """
    send(results, {"type": "ready", "ipc_version": IPC_VERSION, "model": description})

    # Iterating the pipe ends at EOF, which is what a Worker Head that died
    # looks like from here — and means the same thing as `shutdown`.
    for line in commands:
        line = line.strip()
        if not line:
            continue

        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            # Not ours to interpret and not worth dying over.
            print(f"[zeus musibot]: ignoring an unparsable command: {line[:200]!r}")
            continue

        command_type = command.get("type")

        if command_type == "shutdown":
            break

        if command_type == "execute":
            executions = [Execution.from_message(command)]
        elif command_type == "execute-batch":
            executions = [Execution.from_message(one) for one in command.get("executions") or []]
        else:
            # Unknown types are ignored in both directions, so that the protocol
            # can grow without either side breaking.
            continue

        _run_one_command(results, executions, handler)


def _run_one_command(results: TextIO, executions: list[Execution], handler: Handler) -> None:
    """Hand the executions to the handler and guarantee every one is reported."""
    reporter = Reporter(results)

    try:
        handler(executions, reporter)
    except Exception as exception:  # noqa: BLE001
        # Deliberately blind. Whatever went wrong, these executions have to be
        # reported as failed rather than taking the process down with them: a
        # model that dies fails its work anyway, but reports nothing useful,
        # while this error string reaches the Pipeline Execution log.
        for execution in executions:
            reporter.failed(execution.execution_id, str(exception) or type(exception).__name__)

    # A handler that returned without reporting something — through a bug, or
    # an early return — would otherwise leave the head waiting for an answer
    # that is never coming, tying up the worker until the pipeline times out.
    for execution in executions:
        if execution.execution_id not in reporter.reported:
            reporter.failed(
                execution.execution_id,
                "The model finished without reporting this execution.",
            )
