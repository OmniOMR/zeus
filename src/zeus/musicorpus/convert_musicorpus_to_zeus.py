from pathlib import Path
from .extract_samples_for_page import extract_samples_for_page
import json
from ..data.SamplesFile import SamplesFile
import tqdm
from .convert_musicxml import convert_musicxml
from .convert_image import convert_image
import logging


def convert_musicorpus_to_zeus(
        input_path: Path,
        output_path: Path,
        take_staves: bool,
        take_grandstaves: bool,
        re_crop: bool,
        normalize_image_height: int | None,
):
    """
    Converts a MusiCorpus dataset to a Zeus dataset.
    
    :param input_path: Path to a MusiCorpus dataset, e.g. 'CVC.Dolores'
    :param output_path: Path to the output folder, e.g. 'dolores-training'.
        The output folder must not exist before running this function.
    :param take_staves: Include staves in the output.
    :param take_grandstaves: Include grandstaves in the output.
    :param re_crop: Crops sample images from page-level images,
        instead of using crops from the input MusiCorpus dataset.
    :param normalize_image_height: Rescale all sample images to the given height.
    """
    
    # create the output folder
    output_path.mkdir(parents=True, exist_ok=False)

    # load splits
    with open(input_path / "splits.json", "r") as f:
        splits: dict[str, list[str]] = json.load(f)
    
    # define the "all" split
    splits["all"] = list(sorted(set(
        page_name
        for split in splits.values()
        for page_name in split
    )))

    # define the zeus samples files
    samples_files = {
        split_name: SamplesFile.empty(output_path / f"samples.{split_name}.txt")
        for split_name in splits.keys()
    }

    # convert data page by page
    for page_name in tqdm.tqdm(iterable=splits["all"], unit="pages"):
        for mc_sample in extract_samples_for_page(
            page_path=input_path / page_name,
            take_staves=take_staves,
            take_grandstaves=take_grandstaves
        ):
            try:
                convert_musicxml(
                    output_path=output_path,
                    mc_sample=mc_sample,
                )
                convert_image(
                    output_path=output_path,
                    mc_sample=mc_sample,
                    re_crop=re_crop,
                    normalize_image_height=normalize_image_height,
                )
            except:
                # skip the sample on error
                logging.exception(f"Error converting sample {mc_sample.get_zeus_sample_name()}:")
                continue

            # add the sample to the proper zeus samples files
            for split_name, samples_file in samples_files.items():
                if page_name in splits[split_name]:
                    samples_file.append(mc_sample.get_zeus_sample_name())

    # write all samples files
    for samples_file in samples_files.values():
        samples_file.write()

    # print statistics
    print()
    print("These are sample counts for individual splits:")
    for split_name, samples_file in samples_files.items():
        print(split_name + ":", len(samples_file))
    
