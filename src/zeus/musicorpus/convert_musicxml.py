from pathlib import Path, PosixPath
from .MusicorpusSample import MusicorpusSample
from lmx.tokenization.Encoder import Encoder
from lmx.musicxml.io.parse_musicxml_tree_from_string \
    import parse_musicxml_tree_from_string
import sys


def convert_musicxml(
        output_path: Path,
        mc_sample: MusicorpusSample,
):
    """
    This method is responsible for converting MusicXML data
    from MusiCorpus to Zeus for a single sample. It does all
    the looking up, converting and writing of all files.
    It also includes conversion to LMX.
    
    :param output_path: Path to the output Zeus dataset folder.
    :param mc_sample: Description of the sample to be converted.
    """

    # locate the input MusicXML file
    input_musicxml_path = mc_sample.musicorpus_path / "transcription.musicxml"
    assert input_musicxml_path.exists(), \
        f"There is a missing MusicXML file at {input_musicxml_path}"
    
    # load input musicxml string
    input_musicxml_string = input_musicxml_path.read_text("utf-8")

    # NAIVE IMPLEMENTATION: do no transformation or normalization
    output_musicxml_string = input_musicxml_string
    print("WARNING: No MusicXML clef and key normalization is implemented yet!")

    # generate LMX
    lmx_encoder = Encoder(errout=sys.stderr)
    musicxml_tree = parse_musicxml_tree_from_string(output_musicxml_string)
    part_elements = musicxml_tree.findall("part")
    assert len(part_elements) == 1, \
        f"Too many parts in the MusicXML file at {input_musicxml_path}"
    lmx_encoder.process_part(part_elements[0])
    output_lmx_string = " ".join(lmx_encoder.output_tokens)

    # path to the output sample files, without suffix
    output_sample_path = output_path \
        / Path(PosixPath(mc_sample.get_zeus_sample_name()))

    # write output files
    output_sample_path.parent.mkdir(parents=True, exist_ok=True)
    output_sample_path.with_suffix(".musicxml").write_text(output_musicxml_string, "utf-8")
    output_sample_path.with_suffix(".lmx").write_text(output_lmx_string, "utf-8")
