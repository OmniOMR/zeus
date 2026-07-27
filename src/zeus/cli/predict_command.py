import argparse
import os
import sys
from pathlib import Path

NAME = "predict"

DESCRIPTION = "Reads music notation off images and writes MusicXML transcriptions"


def define_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        metavar="IMAGE",
        help="Image files to transcribe, one staff or grandstaff each. "
        + "PNG and JPEG both work.",
    )
    parser.add_argument(
        "--model-snapshot",
        required=True,
        type=Path,
        help="Path to a trained model folder, e.g. 'models/solo26.model'",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Folder to write transcriptions into. By default each "
        + "transcription is written next to the image it came from.",
    )
    parser.add_argument(
        "--lmx",
        default=False,
        action="store_true",
        help="Also write the raw LMX token string beside each MusicXML file",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Number of images per batch when doing inference, defaults to 16",
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
    from ..musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml

    # prepare CLI arguments
    image_paths: list[Path] = list(args.images)
    snapshot_path = Path(args.model_snapshot)
    output_dir: Path | None = Path(args.output_dir) if args.output_dir else None
    also_write_lmx = bool(args.lmx)
    batch_size = int(args.batch_size)

    # Checked before the model is loaded, which takes seconds, so that a typo
    # in a path is reported immediately rather than after the wait.
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        print("There is no file at:")
        for path in missing:
            print(" ", path)
        sys.exit(2)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    images = [path.read_bytes() for path in image_paths]

    zeus = Zeus.load(snapshot_path)
    predictions = zeus.predict(
        images=images,
        inference_options=InferenceOptions(batch_size=batch_size),
        with_progress_bar=True,
    )

    failures = 0
    for image_path, lmx in zip(image_paths, predictions, strict=True):
        destination = output_dir if output_dir is not None else image_path.parent

        if also_write_lmx:
            # Written before the conversion, so that a prediction the decoder
            # rejects can still be looked at — which is when you most want it.
            lmx_path = destination / (image_path.stem + ".lmx")
            lmx_path.write_text(lmx + "\n", encoding="utf-8")
            print("Wrote", lmx_path)

        try:
            musicxml = lmx_to_musicxml(lmx)
        except LmxDecodingError as error:
            # One unreadable staff out of a hundred should cost that staff, not
            # the whole run.
            print(f"Failed on {image_path}: {error}", file=sys.stderr)
            failures += 1
            continue

        musicxml_path = destination / (image_path.stem + ".musicxml")
        musicxml_path.write_text(musicxml, encoding="utf-8")
        print("Wrote", musicxml_path)

    if failures:
        print(f"\n{failures} of {len(image_paths)} images could not be transcribed.")
        sys.exit(1)
