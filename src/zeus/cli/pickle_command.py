import argparse
from pathlib import Path
from ..data.ZeusDataset import ZeusDataset


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "samples_file_path",
        type=str,
        help="Path to the samples.txt file that will be pickled"
    )
    parser.add_argument(
        "--image_suffix",
        default="",
        type=str,
        help="Suffix to add to image base paths before the extension; " +
            "used to create Camera-GrandStaff pickles by adding the " +
            "'_distorted' suffix."
    )
    parser.add_argument(
        "--with_musicxml",
        default=False,
        action="store_true",
        help="Include MusicXML data in the pickle, only needed " +
        "for TEDn evaluation."
    )


def execute(parser: argparse.ArgumentParser, args: argparse.Namespace):
    samples_path = Path(args.samples_file_path)
    image_suffix = str(args.image_suffix)
    with_musicxml = bool(args.with_musicxml)
    
    if not samples_path.exists():
        print("There is no file at", samples_path)
        exit(3)
    
    dataset = ZeusDataset.load_from_samples_file(
        samples_file_path=samples_path,
        image_suffix=image_suffix,
        with_musicxml=with_musicxml,
        show_progress_bar=True,
    )
    
    pickle_path = create_pickle_path_from_samples_path(
        samples_path=samples_path,
        image_suffix=image_suffix
    )

    dataset.write_to_pickle_file(pickle_path)


def create_pickle_path_from_samples_path(
        samples_path: Path,
        image_suffix: str
) -> Path:
    stem = samples_path.stem
    stem_parts = list(stem.split("."))
    stem_parts[0] += image_suffix
    new_stem = ".".join(stem_parts)
    return samples_path.with_stem(new_stem).with_suffix(".pickle")
