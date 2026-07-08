import argparse
from pathlib import Path
from ..visualization.render_zeus_dataset_samples \
    import render_zeus_dataset_samples
from lmx.musescore.MuseScore import MuseScore


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "samples_file_path",
        type=str,
        help="Path to the samples.txt file whose samples will be rendered"
    )
    parser.add_argument(
        "--image_suffix",
        default="_rendered",
        type=str,
        help="Suffix to add to output image base paths before the extension; " +
            "analogous to the Camera-GrandStaff samples with the " +
            "'_distorted' suffix."
    )
    parser.add_argument(
        "--render_invisible",
        default=False,
        action="store_true",
        help="Render invisible clefs and signatures in gray, " +
            "useful for data inspection."
    )
    parser.add_argument(
        "--page_width_tenths",
        default=2_500,
        type=int,
        help="How wide of a paper should be used to render each sample. " +
            "A paper too narrow will overflow and fail rendering for that sample. " +
            "This may also be used to control the maximum image width. " +
            "40 tenths is one staff height. The default width is set at a " +
            "reasonable image aspect ratio for OMR models, may be increased " +
            "for debug images, say to 6_000."
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    samples_path = Path(args.samples_file_path)
    image_suffix = str(args.image_suffix)
    render_invisible = bool(args.render_invisible)
    page_width_tenths = int(args.page_width_tenths)
    
    if not samples_path.exists():
        print("There is no file at", samples_path)
        exit(3)
    
    ms = MuseScore.resolve_linux_default()

    render_zeus_dataset_samples(
        ms=ms,
        samples_path=samples_path,
        image_suffix=image_suffix,
        render_invisible=render_invisible,
        batch_size=100,
        page_width_tenths=page_width_tenths,
    )
