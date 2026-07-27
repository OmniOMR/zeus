"""Tests for the command registry and the CLI's naming conventions.

Building the parser imports every command module, and none of them imports
TensorFlow at module level, so this whole file runs in milliseconds. That is
itself worth protecting: a stray top-level `import tensorflow` in a command
module would put five seconds in front of `zeus --help`.
"""

import argparse
import re

import pytest

from zeus.cli.run import COMMANDS, Command, build_parser

KEBAB_CASE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.NAME)
def test_every_command_declares_itself(command: Command) -> None:
    assert isinstance(command.NAME, str) and command.NAME
    assert isinstance(command.DESCRIPTION, str) and command.DESCRIPTION


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.NAME)
def test_command_names_are_kebab_case(command: Command) -> None:
    assert KEBAB_CASE.match(command.NAME)


def test_command_names_are_unique() -> None:
    names = [command.NAME for command in COMMANDS]

    assert len(set(names)) == len(names)


def test_the_parser_builds() -> None:
    assert build_parser() is not None


def collect_option_strings(parser: argparse.ArgumentParser) -> list[str]:
    """Every `--flag` the parser and its subparsers accept."""
    options: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                options.extend(collect_option_strings(subparser))
        options.extend(option for option in action.option_strings if option.startswith("--"))
    return options


def test_no_flag_is_spelled_with_an_underscore() -> None:
    """Flags are kebab-case, as they are everywhere in Musibot.

    argparse turns `--batch-size` into `args.batch_size` by itself, so this is
    purely about what a user types.
    """
    underscored = sorted(
        {option for option in collect_option_strings(build_parser()) if "_" in option}
    )

    assert underscored == []


def test_running_without_a_command_asks_for_one() -> None:
    """`zeus` alone should print help and fail, not traceback or succeed."""
    parser = build_parser()

    args = parser.parse_args([])

    assert args.command is None
