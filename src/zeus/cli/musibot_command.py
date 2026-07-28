import argparse
import os
import sys
from pathlib import Path

NAME = "musibot"

DESCRIPTION = "Runs Zeus as a Musibot model, driven by a worker head over pipes"


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-snapshot",
        required=True,
        type=Path,
        help="Path to the trained model folder to serve, "
        + "e.g. 'models/solo26.model'. The snapshot also says what the model "
        + "reads and what name and version it announces to Musibot.",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Largest number of images fed through the model in one forward "
        + "pass, defaults to 16. A worker head decides how many executions to "
        + "send at once; this bounds what happens to them here.",
    )
    parser.add_argument(
        "--no-lmx",
        default=False,
        action="store_true",
        help="Write only transcription.musicxml, without the " + "transcription.lmx beside it",
    )
    parser.add_argument(
        "--quiet-tf",
        default=False,
        action="store_true",
        help="Set Tensorflow logging to level 2 "
        + "(hides debugging messages, reports only errors)",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.quiet_tf:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # deffered imports as they import tensorflow which is slow
    from ..model.inference_options import InferenceOptions
    from ..model.zeus import Zeus
    from ..musibot.protocol import serve
    from ..musibot.zeus_model import ZeusMusibotModel

    snapshot_path = Path(args.model_snapshot)
    batch_size = int(args.batch_size)
    write_lmx = not bool(args.no_lmx)

    # A Worker Head passes the two descriptors it created and the directory it
    # mirrors pages into. Their absence means this was started by hand, which
    # is worth saying plainly — the failure is otherwise a KeyError.
    missing = [
        variable
        for variable in ("MUSIBOT_IPC_COMMAND_FD", "MUSIBOT_IPC_RESULT_FD", "MUSIBOT_PAGES_DIR")
        if variable not in os.environ
    ]
    if missing:
        print(
            "This command is started by a Musibot worker head, not by hand.\n"
            f"Missing from the environment: {', '.join(missing)}.\n"
            "See docs/musibot-model.md for how a worker head is pointed at it.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not snapshot_path.is_dir():
        print(f"There is no model snapshot at {snapshot_path}", file=sys.stderr)
        sys.exit(2)

    # The protocol runs on two dedicated descriptors rather than on stdin and
    # stdout, which leaves this process free to print: whatever it writes is
    # captured by the worker head as this model's log and reaches the pipeline
    # execution log, and a stray print cannot corrupt the protocol.
    commands = os.fdopen(int(os.environ["MUSIBOT_IPC_COMMAND_FD"]), "r")
    results = os.fdopen(int(os.environ["MUSIBOT_IPC_RESULT_FD"]), "w")
    pages_dir = Path(os.environ["MUSIBOT_PAGES_DIR"])

    # Loaded before `ready` goes out, so that a worker head is offered no work
    # during the seconds this takes.
    model = ZeusMusibotModel(
        zeus=Zeus.load(snapshot_path),
        snapshot_path=snapshot_path,
        pages_dir=pages_dir,
        inference_options=InferenceOptions(batch_size=batch_size),
        write_lmx=write_lmx,
    )

    description = model.describe()
    print(f"zeus musibot: serving {description['name']} {description['version']}")
    print(f"zeus musibot: reads {', '.join(model.model_options.input_subdivisions)}")

    serve(
        commands=commands,
        results=results,
        description=description,
        handler=model.handle,
    )
