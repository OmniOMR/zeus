from pathlib import Path, PosixPath
from .musicorpus_sample import MusicorpusSample
from lmx.tokenization.Encoder import Encoder
from lmx.musicxml.io.read_musicxml_tree_from_file \
    import read_musicxml_tree_from_file
from lmx.musicxml.io.write_musicxml_tree_to_file \
    import write_musicxml_tree_to_file
import sys
from lmx.musicxml.omitted_staff_header.normalize_invisible_header_clef \
    import normalize_invisible_header_clef
from lmx.musicxml.omitted_staff_header.normalize_invisible_key_signature \
    import normalize_invisible_key_signature
from lmx.musicxml.omitted_staff_header.normalize_invisible_time_signature \
    import normalize_invisible_time_signature
from lmx.musicxml.pitch.Clef import G_CLEF, F_CLEF


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
    
    # load input musicxml <part> element
    musicxml_tree = read_musicxml_tree_from_file(input_musicxml_path)
    part_elements = musicxml_tree.findall("part")
    assert len(part_elements) == 1, \
        f"Too many parts in the MusicXML file at {input_musicxml_path}"
    part_element = part_elements[0]

    # determine whether it is a grandstaff
    is_grandstaff = (
        part_element.findtext("measure/attributes/staves", "1") == "2"
    )

    # normalize invisible header
    musicxml_tree.getroot().remove(part_element) # remove the original <part>
    part_element = normalize_invisible_header_clef(
        part_element=part_element,
        desired_clef=[G_CLEF, F_CLEF] if is_grandstaff else G_CLEF,
        when_clef_visible="dont-normalize",
    )
    part_element = normalize_invisible_key_signature(
        part_element=part_element,
        desired_key=0,
        when_key_visible="dont-normalize",
    )
    part_element = normalize_invisible_time_signature(
        part_element=part_element,
        desired_time=None,
        when_time_visible="dont-normalize",
    )
    musicxml_tree.getroot().append(part_element) # re-insert the <part>

    # generate LMX
    lmx_encoder = Encoder(errout=sys.stderr)
    lmx_encoder.process_part(part_element)
    output_lmx_string = " ".join(lmx_encoder.output_tokens)

    # path to the output sample files, without suffix
    output_sample_path = output_path \
        / Path(PosixPath(mc_sample.get_zeus_sample_name()))

    # write output files
    output_sample_path.parent.mkdir(parents=True, exist_ok=True)
    write_musicxml_tree_to_file(
        output_sample_path.with_suffix(".musicxml"),
        musicxml_tree
    )
    output_sample_path.with_suffix(".lmx").write_text(output_lmx_string, "utf-8")
