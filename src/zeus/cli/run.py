import argparse
import sys

import zeus.cli.evaluate_command
import zeus.cli.musicorpus_command
import zeus.cli.pickle_command
import zeus.cli.render_command
import zeus.cli.train_command
import zeus.cli.visualize_data_command
import zeus.cli.visualize_predictions_command

# None of these command modules imports TensorFlow at module level — each
# defers it into its `execute` — so building the whole parser stays cheap and
# `zeus --help` answers immediately.

parser = argparse.ArgumentParser(prog="zeus", description="CLI for using the Zeus model")

subparsers = parser.add_subparsers(title="available commands", dest="root_command_name")

root_command_handlers = {}


############################
# Define all root commands #
############################


# === train ===

zeus.cli.train_command.define_parser(
    subparsers.add_parser(
        "train", aliases=[], description="Trains a new model on the given dataset"
    )
)
root_command_handlers["train"] = zeus.cli.train_command.execute


# === visualize data ===

zeus.cli.visualize_data_command.define_parser(
    subparsers.add_parser(
        "visualize_data",
        aliases=[],
        description="Visualizes training data for given training settings",
    )
)
root_command_handlers["visualize_data"] = zeus.cli.visualize_data_command.execute


# === evaluate ===

zeus.cli.evaluate_command.define_parser(
    subparsers.add_parser(
        "evaluate", aliases=[], description="Evaluates a trained model against a given dataset"
    )
)
root_command_handlers["evaluate"] = zeus.cli.evaluate_command.execute


# === visualize predictions ===

zeus.cli.visualize_predictions_command.define_parser(
    subparsers.add_parser(
        "visualize_predictions",
        aliases=[],
        description="Visualizes predictions that result from the 'evaluate' command",
    )
)
root_command_handlers["visualize_predictions"] = zeus.cli.visualize_predictions_command.execute


# === pickle ===

zeus.cli.pickle_command.define_parser(
    subparsers.add_parser(
        "pickle",
        aliases=[],
        description="Creates a pickled representation of a dataset "
        + "represented by samples.txt file. Creates a "
        + "samples.pickle file that can be used by Zeus.",
    )
)
root_command_handlers["pickle"] = zeus.cli.pickle_command.execute


# === predict ===

# TODO


# === musicorpus ===

zeus.cli.musicorpus_command.define_parser(
    subparsers.add_parser(
        "musicorpus", aliases=[], description="Converts a MusiCorpus dataset into a Zeus dataset"
    )
)
root_command_handlers["musicorpus"] = zeus.cli.musicorpus_command.execute


# === render ===

zeus.cli.render_command.define_parser(
    subparsers.add_parser(
        "render", aliases=[], description="Renders Zeus dataset MusicXML samples via MuseScore"
    )
)
root_command_handlers["render"] = zeus.cli.render_command.execute


######################
# Execute the parser #
######################


def run():
    """
    This method is called from all the places
    that need a method reference to invoke this CLI
    """
    args = parser.parse_args()

    if args.root_command_name is None:
        parser.print_help()
        sys.exit(2)

    root_command_handlers[args.root_command_name](parser, args)


if __name__ == "__main__":
    run()
