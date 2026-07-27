import argparse
import shutil
import sys
from pathlib import Path

from ..musicorpus.convert_musicorpus_to_zeus import convert_musicorpus_to_zeus

NAME = "musicorpus"

DESCRIPTION = "Converts a MusiCorpus dataset into a Zeus dataset"


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input MusiCorpus dataset, e.g. path to the 'CVC.Dolores' folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output Zeus dataset, e.g. path to the 'dolores-training' folder",
    )
    parser.add_argument(
        "--take-staves", default=False, action="store_true", help="Take all solo staves as samples"
    )
    parser.add_argument(
        "--take-grandstaves",
        default=False,
        action="store_true",
        help="Take all grandstaves as samples",
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Overwrite the output folder if it already exists",
    )
    parser.add_argument(
        "--re-crop",
        default=False,
        action="store_true",
        help="Manually crop sample images from the page images, instead of "
        + "using crops from the MusiCorpus dataset. Not implemented yet.",
    )
    parser.add_argument(
        "--normalize-image-height",
        type=str,
        default=None,
        help="Rescale sample images to the given height in pixels. Not implemented yet.",
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    input_path = Path(args.input)
    output_path = Path(args.output)
    take_staves = bool(args.take_staves)
    take_grandstaves = bool(args.take_grandstaves)
    force = bool(args.force)
    re_crop = bool(args.re_crop)
    normalize_image_height = (
        None if args.normalize_image_height is None else int(args.normalize_image_height)
    )

    if not input_path.is_dir():
        print("There is no folder at", input_path)
        sys.exit(2)

    if not take_staves and not take_grandstaves:
        print(
            "You must at least --take_staves or --take_grandstaves or both, "
            + "but taking none would produce empty output dataset."
        )
        sys.exit(4)

    # Both of these were checked deep inside the per-sample conversion, which
    # called `exit(0)` on the first sample it reached — reporting success while
    # leaving a half-written output folder behind. They are properties of the
    # request, so they are refused here, before anything is created.
    if re_crop:
        print("Re-cropping is not yet implemented.")
        sys.exit(5)

    if normalize_image_height is not None:
        print("Image height normalization is not yet implemented.")
        sys.exit(6)

    if output_path.exists():
        if force:
            shutil.rmtree(output_path)
        else:
            print("The output folder already exists, use --force to overwrite it.")
            sys.exit(3)

    convert_musicorpus_to_zeus(
        input_path=input_path,
        output_path=output_path,
        take_staves=take_staves,
        take_grandstaves=take_grandstaves,
        re_crop=re_crop,
        normalize_image_height=normalize_image_height,
    )
