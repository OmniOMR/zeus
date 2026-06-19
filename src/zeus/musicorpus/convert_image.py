from pathlib import Path, PosixPath
from .MusicorpusSample import MusicorpusSample


def convert_image(
        output_path: Path,
        mc_sample: MusicorpusSample,
        re_crop: bool,
        normalize_image_height: int | None,
):
    """
    This method is responsible for converting image data
    from MusiCorpus to Zeus for a single sample. It does all
    the looking up, converting and writing of all files.
    
    :param output_path: Path to the output Zeus dataset folder.
    :param mc_sample: Description of the sample to be converted.
    :param re_crop: Crops sample images from page-level images,
        instead of using crops from the input MusiCorpus dataset.
    :param normalize_image_height: Rescale all sample images to the given height.
    """

    if re_crop:
        print("Re-cropping is not yet implemented")
        exit(0)
    
    if normalize_image_height is not None:
        print("Image height normalization is not yet implemented")
        exit(0)
    
    # locate the input image file
    input_image_path = mc_sample.musicorpus_path / "image.jpg"
    assert input_image_path.exists(), \
        f"There is a missing image file at {input_image_path}"
    
    # load the image bytes
    input_image_bytes = input_image_path.read_bytes()

    # NAIVE IMPLEMENTATION: do no transformation
    output_image_bytes = input_image_bytes

    # path to the output sample files, without suffix
    output_sample_path = output_path \
        / Path(PosixPath(mc_sample.get_zeus_sample_name()))

    # write output file
    output_sample_path.parent.mkdir(parents=True, exist_ok=True)
    output_sample_path.with_suffix(".jpg").write_bytes(output_image_bytes)