"""Turning the model's LMX output into MusicXML.

The model predicts LMX — a linearized, token-per-symbol encoding of music
notation — because that is what an image-to-sequence model can emit. Everything
downstream of Zeus wants MusicXML, so this is the last step of every prediction
path, whether it came from the CLI or from a Musibot worker.

The conversion itself belongs to the `lmx` package; what this module adds is a
single entry point with one opinion about error handling, so that both callers
behave the same way.
"""

import io
from typing import TextIO


class LmxDecodingError(Exception):
    """Raised when a predicted LMX string cannot be turned into MusicXML.

    A model's output is a prediction, not a promise: nothing constrains it to
    be well-formed LMX, and a confused model emits token sequences the decoder
    cannot make sense of. That is an ordinary outcome for one sample and must
    not take down a run of a hundred, so it is raised as one named exception
    that callers can catch per sample rather than as whatever the decoder
    happened to hit.
    """


def lmx_to_musicxml(lmx: str, errout: TextIO | None = None) -> str:
    """Convert one LMX token string into a MusicXML document.

    :param lmx: Whitespace-separated LMX tokens, as `Zeus.predict` returns
        them.
    :param errout: Where the decoder reports tokens it could not make sense of.
        `None` discards them, which is the right default for a prediction: a
        model's output is not required to be well-formed, and one confused
        token should not cost the whole transcription.
    :returns: A complete MusicXML document, with the part wrapped in a
        `score-partwise` score, as a string.
    """
    # Imported here rather than at module level to keep the import graph of the
    # CLI light; `lmx` is not expensive, but nothing here is needed until a
    # prediction actually has to be written out.
    from lmx.musicxml.io.serialize_musicxml_tree_to_string import (
        serialize_musicxml_tree_to_string,
    )
    from lmx.musicxml.part_to_score import part_to_score
    from lmx.tokenization.Decoder import Decoder

    decoder = Decoder(errout=errout if errout is not None else io.StringIO())

    try:
        decoder.process_text(lmx)
        return serialize_musicxml_tree_to_string(part_to_score(decoder.part_element))
    except Exception as exception:
        # Deliberately broad. The decoder reports most confusion by writing to
        # `errout` and carrying on, but not all of it: some malformed input
        # reaches code that assumes well-formed input and fails with whatever
        # that code fails with. As of the pinned `lmx` commit, an LMX string
        # with tokens before its first `measure` raises `AttributeError` from
        # inside the decoder's own error-reporting path — an exception that
        # says nothing about the real problem to whoever has to read it.
        raise LmxDecodingError(
            f"Could not decode the predicted LMX into MusicXML: {exception}. "
            f"The LMX was: {lmx[:200]!r}"
        ) from exception
