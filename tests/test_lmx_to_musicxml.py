"""Tests for turning the model's LMX output into MusicXML.

No model and no TensorFlow here: the conversion takes a token string, so it can
be tested with token strings.
"""

import xml.etree.ElementTree as ET

import pytest

from zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml

A_SIMPLE_MEASURE = "measure key:fifths:0 time beats:4 beat-type:4 clef:G2 C4 quarter"


def find(element: ET.Element, path: str) -> ET.Element:
    """`Element.find` returning the element, or failing the test saying so."""
    found = element.find(path)
    assert found is not None, f"no {path!r} in the produced MusicXML"
    return found


def test_well_formed_lmx_becomes_well_formed_musicxml() -> None:
    musicxml = lmx_to_musicxml(A_SIMPLE_MEASURE)

    root = ET.fromstring(musicxml)
    assert root.tag == "score-partwise"
    assert find(root, "part") is not None
    assert find(root, "part-list/score-part/part-name") is not None


def test_the_notes_survive_the_trip() -> None:
    musicxml = lmx_to_musicxml(A_SIMPLE_MEASURE)

    root = ET.fromstring(musicxml)
    measures = find(root, "part").findall("measure")
    assert len(measures) == 1

    pitch = find(measures[0], "note/pitch")
    assert find(pitch, "step").text == "C"
    assert find(pitch, "octave").text == "4"


def test_several_measures_stay_several_measures() -> None:
    lmx = " ".join([A_SIMPLE_MEASURE, "measure C4 quarter", "measure C4 quarter"])

    root = ET.fromstring(lmx_to_musicxml(lmx))

    assert len(find(root, "part").findall("measure")) == 3


def test_a_prediction_the_decoder_cannot_read_raises_our_own_error() -> None:
    """A model's output is a prediction, not a promise.

    The `lmx` package raises `AttributeError` from inside its own error
    reporting for this input, which tells a caller nothing. Callers catch
    `LmxDecodingError` and report the sample as failed, so it has to be what
    comes out.
    """
    with pytest.raises(LmxDecodingError, match="Could not decode the predicted LMX"):
        lmx_to_musicxml("A2 E2 A2")


def test_the_error_carries_the_offending_lmx() -> None:
    """So that a failure in a batch of a hundred can be looked into."""
    with pytest.raises(LmxDecodingError, match="C9 D9 E9"):
        lmx_to_musicxml("C9 D9 E9")


def test_empty_lmx_does_not_crash() -> None:
    """The model can predict nothing at all, and that is an empty score."""
    root = ET.fromstring(lmx_to_musicxml(""))

    assert root.tag == "score-partwise"
