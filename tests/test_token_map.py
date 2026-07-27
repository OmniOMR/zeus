"""Tests for the map between LMX tokens and model output indices.

The token map is the one part of a snapshot that is text rather than weights,
and getting it wrong corrupts every prediction silently — a shifted index
decodes to a real but wrong token. These tests are cheap because nothing here
imports TensorFlow.
"""

from pathlib import Path

import pytest

from zeus.model.token_map import TokenMap


def a_token_map() -> TokenMap:
    """A small map shaped like a real one: BOS/EOS first, unknown second."""
    return TokenMap(
        tokens=["<bos/eos>", "<unk>", "measure", "C4", "quarter"],
        bos_token_index=0,
        eos_token_index=0,
        unknown_token_index=1,
    )


def test_tokens_and_indices_round_trip() -> None:
    token_map = a_token_map()

    for index, token in enumerate(token_map.tokens):
        assert token_map.token_to_index(token) == index
        assert token_map.index_to_token(index) == token


def test_unknown_token_is_refused_unless_allowed() -> None:
    token_map = a_token_map()

    with pytest.raises(Exception, match="not known to the model"):
        token_map.token_to_index("B-flat-major")

    assert token_map.token_to_index("B-flat-major", allow_unknown_tokens=True) == 1


def test_index_out_of_range_is_refused() -> None:
    token_map = a_token_map()

    with pytest.raises(Exception, match="out of range"):
        token_map.index_to_token(len(token_map))


def test_decoding_drops_the_boundary_tokens() -> None:
    """BOS and EOS are the model's own bookkeeping and are not music."""
    token_map = a_token_map()

    lmx = token_map.indices_to_lmx([0, 2, 3, 4, 0])

    assert lmx == "measure C4 quarter"


def test_length_is_the_size_of_the_output_layer() -> None:
    assert len(a_token_map()) == 5


def test_duplicate_tokens_are_refused() -> None:
    """Two indices decoding to one token would make the map non-invertible."""
    with pytest.raises(AssertionError, match="duplicates"):
        TokenMap(
            tokens=["<bos/eos>", "<unk>", "C4", "C4"],
            bos_token_index=0,
            eos_token_index=0,
            unknown_token_index=1,
        )


def test_a_map_survives_a_trip_through_a_snapshot(tmp_path: Path) -> None:
    written = a_token_map()
    written.write_to_model_folder(tmp_path)

    loaded = TokenMap.load_from_model_folder(tmp_path)

    assert loaded.tokens == written.tokens
    assert loaded.bos_token_index == written.bos_token_index
    assert loaded.unknown_token_index == written.unknown_token_index


def test_a_2024_snapshot_gains_the_boundary_token_it_never_stored(
    tmp_path: Path,
) -> None:
    """The 2024 models kept BOS/EOS inside the Keras model rather than in the
    vocabulary file, and shifted every index by one to make room for it. Loading
    one now has to put that token back, or every prediction decodes off by one.
    """
    (tmp_path / "tags.txt").write_text("<unk>\nmeasure\nC4\n")

    token_map = TokenMap.load_from_model_folder(tmp_path)

    assert token_map.tokens == ["<bos/eos>", "<unk>", "measure", "C4"]
    assert token_map.token_to_index("measure") == 2
