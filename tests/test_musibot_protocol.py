"""Tests for the worker IPC transport.

The loop is driven over in-memory streams instead of real pipes, so these need
neither a worker head nor a subprocess nor a model — and no TensorFlow, so they
run in milliseconds.
"""

import io
import json
from typing import Any

from zeus.musibot.protocol import IPC_VERSION, Execution, Reporter, serve

A_DESCRIPTION = {
    "name": "zeus-test",
    "version": "1.0.0",
    "signature": {"input": ["Staves/1/image.jpg"], "output": ["Staves/1/transcription.musicxml"]},
    "supports_batching": True,
}


def commands_from(*messages: dict[str, Any]) -> io.StringIO:
    return io.StringIO("".join(json.dumps(message) + "\n" for message in messages))


def messages_in(results: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in results.getvalue().splitlines() if line.strip()]


def an_execution(execution_id: str = "e1", page: str = "PAGE") -> dict[str, Any]:
    return {
        "type": "execute",
        "execution_id": execution_id,
        "page": page,
        "input": ["Staves/1/image.jpg"],
        "parameters": {},
    }


def complete_everything(executions: list[Execution], reporter: Reporter) -> None:
    for execution in executions:
        reporter.completed(execution.execution_id)


def test_it_announces_itself_before_anything_else() -> None:
    results = io.StringIO()

    serve(commands_from(), results, A_DESCRIPTION, complete_everything)

    ready = messages_in(results)[0]
    assert ready["type"] == "ready"
    assert ready["ipc_version"] == IPC_VERSION
    assert ready["model"] == A_DESCRIPTION


def test_an_execution_is_reported_completed() -> None:
    results = io.StringIO()

    serve(commands_from(an_execution()), results, A_DESCRIPTION, complete_everything)

    assert messages_in(results)[1] == {"type": "completed", "execution_id": "e1"}


def test_a_batch_reports_each_execution_separately() -> None:
    results = io.StringIO()
    batch = {
        "type": "execute-batch",
        "executions": [an_execution("e1"), an_execution("e2"), an_execution("e3", page="OTHER")],
    }

    serve(commands_from(batch), results, A_DESCRIPTION, complete_everything)

    reported = [m for m in messages_in(results) if m["type"] == "completed"]
    assert [m["execution_id"] for m in reported] == ["e1", "e2", "e3"]


def test_one_failure_in_a_batch_does_not_fail_its_batch_mates() -> None:
    results = io.StringIO()
    batch = {"type": "execute-batch", "executions": [an_execution("e1"), an_execution("e2")]}

    def fail_the_first(executions: list[Execution], reporter: Reporter) -> None:
        reporter.failed(executions[0].execution_id, "no staves found")
        reporter.completed(executions[1].execution_id)

    serve(commands_from(batch), results, A_DESCRIPTION, fail_the_first)

    reports = {m["execution_id"]: m for m in messages_in(results) if "execution_id" in m}
    assert reports["e1"]["type"] == "failed"
    assert reports["e1"]["error"] == "no staves found"
    assert reports["e2"]["type"] == "completed"


def test_a_handler_that_raises_fails_its_executions_rather_than_the_process() -> None:
    """A model that dies fails its work anyway but reports nothing useful."""
    results = io.StringIO()

    def explode(executions: list[Execution], reporter: Reporter) -> None:
        raise RuntimeError("the GPU fell over")

    serve(commands_from(an_execution()), results, A_DESCRIPTION, explode)

    report = messages_in(results)[1]
    assert report["type"] == "failed"
    assert "the GPU fell over" in report["error"]


def test_an_execution_the_handler_forgot_is_still_reported() -> None:
    """Otherwise the head waits for an answer that is never coming, tying up
    the worker until the pipeline times out."""
    results = io.StringIO()

    def report_nothing(executions: list[Execution], reporter: Reporter) -> None:
        return

    serve(commands_from(an_execution()), results, A_DESCRIPTION, report_nothing)

    report = messages_in(results)[1]
    assert report["type"] == "failed"
    assert "without reporting" in report["error"]


def test_an_execution_is_not_reported_twice() -> None:
    results = io.StringIO()

    def report_twice(executions: list[Execution], reporter: Reporter) -> None:
        reporter.completed(executions[0].execution_id)
        reporter.failed(executions[0].execution_id, "changed my mind")

    serve(commands_from(an_execution()), results, A_DESCRIPTION, report_twice)

    reports = [m for m in messages_in(results) if m.get("execution_id") == "e1"]
    assert len(reports) == 1
    assert reports[0]["type"] == "completed"


def test_shutdown_stops_the_loop() -> None:
    results = io.StringIO()

    serve(
        commands_from({"type": "shutdown"}, an_execution("never-run")),
        results,
        A_DESCRIPTION,
        complete_everything,
    )

    assert [m["type"] for m in messages_in(results)] == ["ready"]


def test_end_of_the_command_pipe_means_stop() -> None:
    """It is what the model sees if the worker head dies."""
    results = io.StringIO()

    serve(commands_from(an_execution()), results, A_DESCRIPTION, complete_everything)

    assert [m["type"] for m in messages_in(results)] == ["ready", "completed"]


def test_unknown_message_types_are_ignored() -> None:
    """So that the protocol can grow without either side breaking."""
    results = io.StringIO()

    serve(
        commands_from({"type": "sing-a-song"}, an_execution()),
        results,
        A_DESCRIPTION,
        complete_everything,
    )

    assert [m["type"] for m in messages_in(results)] == ["ready", "completed"]


def test_an_unparsable_line_does_not_stop_the_loop() -> None:
    results = io.StringIO()
    commands = io.StringIO("not json at all\n" + json.dumps(an_execution()) + "\n")

    serve(commands, results, A_DESCRIPTION, complete_everything)

    assert [m["type"] for m in messages_in(results)] == ["ready", "completed"]


def test_progress_can_be_attributed_to_one_execution() -> None:
    results = io.StringIO()

    def report_progress(executions: list[Execution], reporter: Reporter) -> None:
        reporter.progress(executions[0].execution_id, "staff 3/12", fraction=0.25)
        reporter.completed(executions[0].execution_id)

    serve(commands_from(an_execution()), results, A_DESCRIPTION, report_progress)

    progress = messages_in(results)[1]
    assert progress == {
        "type": "progress",
        "execution_id": "e1",
        "message": "staff 3/12",
        "fraction": 0.25,
    }


def test_every_message_is_flushed_as_it_is_written() -> None:
    """An unflushed message is not late but invisible: a pipe is
    block-buffered, and the head would wait forever."""
    flushes: list[int] = []

    class CountingStream(io.StringIO):
        def flush(self) -> None:
            flushes.append(len(self.getvalue()))
            super().flush()

    results = CountingStream()
    serve(commands_from(an_execution()), results, A_DESCRIPTION, complete_everything)

    # One per message written, each after the message is in the buffer.
    assert len(flushes) == 2
    assert flushes == sorted(flushes)


def test_an_execution_carries_its_parameters_through() -> None:
    seen: list[Execution] = []

    def remember(executions: list[Execution], reporter: Reporter) -> None:
        seen.extend(executions)
        complete_everything(executions, reporter)

    command = an_execution()
    command["parameters"] = {"write_lmx": True}
    serve(commands_from(command), io.StringIO(), A_DESCRIPTION, remember)

    assert seen[0].parameters == {"write_lmx": True}
    assert seen[0].page == "PAGE"
    assert seen[0].input == ["Staves/1/image.jpg"]
