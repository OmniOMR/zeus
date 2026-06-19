import argparse
from pathlib import Path
import shutil
from ..musicorpus.convert_musicorpus_to_zeus import convert_musicorpus_to_zeus


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input MusiCorpus dataset, e.g. path to the 'CVC.Dolores' folder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output Zeus dataset, e.g. path to the 'dolores-training' folder"
    )
    parser.add_argument(
        "--take_staves",
        default=False,
        action="store_true",
        help="Take all solo staves as samples"
    )
    parser.add_argument(
        "--take_grandstaves",
        default=False,
        action="store_true",
        help="Take all grandstaves as samples"
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Overwrite the output folder if it already exists"
    )
    parser.add_argument(
        "--re_crop",
        default=False,
        action="store_true",
        help="Manually crop sample images from the page images, instead of using crops from the MusiCorpus dataset"
    )
    parser.add_argument(
        "--normalize_image_height",
        type=str,
        default=None,
        help="Rescale sample images to the given height in pixels"
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    input_path = Path(args.input)
    output_path = Path(args.output)
    take_staves = bool(args.take_staves)
    take_grandstaves = bool(args.take_grandstaves)
    force = bool(args.force)
    re_crop = bool(args.re_crop)
    normalize_image_height = None if args.normalize_image_height is None \
        else int(args.normalize_image_height)
    
    if not input_path.is_dir():
        print("There is no folder at", input_path)
        exit(2)
    
    if output_path.exists():
        if force:
            shutil.rmtree(output_path)
        else:
            print("The output folder already exists, use --force to overwrite it.")
            exit(3)
    
    if not take_staves and not take_grandstaves:
        print("You must at least --take_staves or --take_grandstaves or both, but taking none would produce empty output dataset.")
        exit(4)
    
    convert_musicorpus_to_zeus(
        input_path=input_path,
        output_path=output_path,
        take_staves=take_staves,
        take_grandstaves=take_grandstaves,
        re_crop=re_crop,
        normalize_image_height=normalize_image_height,
    )
