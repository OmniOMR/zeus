"""The `zeus` command line tool.

Each command lives in a module of its own that declares its own `NAME` and
`DESCRIPTION` and provides `define_parser` and `execute`. This module only
collects them; adding a command means writing that module and naming it in
`COMMANDS` below.

None of the command modules imports TensorFlow at module level — each defers it
into its `execute` — so building the whole parser stays cheap and `zeus --help`
answers immediately.
"""

import argparse
import sys
from typing import Protocol

from zeus.cli import (
    evaluate_command,
    musibot_command,
    musicorpus_command,
    pickle_command,
    predict_command,
    render_command,
    train_command,
    visualize_data_command,
    visualize_predictions_command,
)


class Command(Protocol):
    """What a command module has to provide to appear in `COMMANDS`."""

    NAME: str
    """The subcommand as typed, e.g. `visualize-data`."""

    DESCRIPTION: str
    """One line, shown both in `zeus --help` and in the command's own help."""

    def define_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add this command's arguments to its subparser."""
        ...

    def execute(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        """Run the command."""
        ...


COMMANDS: list[Command] = [
    train_command,
    evaluate_command,
    predict_command,
    musibot_command,
    visualize_data_command,
    visualize_predictions_command,
    pickle_command,
    musicorpus_command,
    render_command,
]


def build_parser() -> argparse.ArgumentParser:
    """Assemble the whole CLI, one subparser per command."""
    parser = argparse.ArgumentParser(prog="zeus", description="CLI for using the Zeus model")
    subparsers = parser.add_subparsers(title="available commands", dest="command")

    for command in COMMANDS:
        command.define_parser(
            subparsers.add_parser(
                command.NAME,
                # `description` heads the command's own --help; `help` is the
                # line beside its name in `zeus --help`, which listed bare
                # command names and no explanation before.
                description=command.DESCRIPTION,
                help=command.DESCRIPTION,
            )
        )

    return parser


def run() -> None:
    """The `zeus` entry point, and what `python -m zeus` calls."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    for command in COMMANDS:
        if args.command == command.NAME:
            command.execute(parser, args)
            return

    raise AssertionError(f"argparse accepted a command nobody handles: {args.command!r}")


if __name__ == "__main__":
    run()
