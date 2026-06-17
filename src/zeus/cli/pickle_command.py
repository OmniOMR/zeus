import argparse
from ..Samples import Samples
from pathlib import Path
from ..RawDatasetSample import RawDatasetSample
import tqdm


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
    if not samples_path.exists():
        print("There is no file at", samples_path)
        exit(3)
    
    image_suffix = str(args.image_suffix)
    with_musicxml = bool(args.with_musicxml)
    
    raw_dataset_samples: list[RawDatasetSample] = []
    
    samples = Samples.load(samples_path)
    with tqdm.tqdm(total=len(samples)) as pbar:
        for sample in samples:

            # load image
            image: bytes | None = None
            for extension in [".jpg", ".png"]:
                image_path = sample.path \
                    .with_name(sample.path.name + image_suffix) \
                    .with_suffix(extension)
                if image_path.exists():
                    image = image_path.read_bytes()
                    break
            if image is None:
                print("Couldn't find image for sample", sample.name)
                exit(4)

            # load lmx
            lmx = sample.path.with_suffix(".lmx") \
                .read_text() \
                .rstrip("\r\n")

            # load musicxml
            musicxml: str | None = None
            if with_musicxml:
                musicxml = sample.path \
                    .with_suffix(".musicxml") \
                    .read_text()
            
            raw_dataset_samples.append(RawDatasetSample(
                sample_name=sample.name,
                image=image,
                lmx=lmx,
                musicxml=musicxml,
            ))

            pbar.update(1)
    
    # pickle raw samples
    pickle_path = create_pickle_path_from_samples_path(
        samples_path=samples_path,
        image_suffix=image_suffix
    )
    RawDatasetSample.write_samples(
        pickle_path=pickle_path,
        samples=raw_dataset_samples,
    )


def create_pickle_path_from_samples_path(
        samples_path: Path,
        image_suffix: str
) -> Path:
    stem = samples_path.stem
    stem_parts = list(stem.split("."))
    stem_parts[0] += image_suffix
    new_stem = ".".join(stem_parts)
    return samples_path.with_stem(new_stem).with_suffix(".pickle")
