# MusicXML, LMX and tokenization

An image-to-sequence model cannot output MusicXML directly. It needs a tokenizer for encoding and for robust and error-resiliant decoding. Zeus is built around [Linearized MusicXML](https://github.com/OMR-Research/lmx) (LMX) - a tokenized representation of MusicXML, which also comes with the tokenizer implementation.

A Zeus model (the neural network) lives only in the space of LMX tokens and knows nothing about the underlying MusicXML. This poses a couple of friction points:

- Token vocabulary
- LMX versioning
- LMX/MusicXML semantics

These are explained more in the following sections.


## Token vocabulary

When a model is trained, it only needs to know about LMX tokens that are present in its training dataset. Any additional tokens don't need to be added to the output layer softmax, becuase they will never be outputted by the model anyways.

Each model snapshot comes with a `TokenMap` (see the Python class), a mapping from output layer neuron indexes to specific LMX token names (strings). This map is an inseparable part of the model, otherwise the LMX semantics connection gets lost.

This has implications when fine-tuning an existing model - it may encounter unknown tokens in its finetuning data. A `TokenMap` resolves this by having a special token `unknown`, which it uses to encode all unkown tokens. This technical problem has implications - an old model with limited LMX vocabulary (say without tuplets) cannot be fine-tuned or re-trained to learn to output these new vocabulary tokens. A new model has to be trained from scratch instead.


## LMX versioning

LMX currently supports a reasonable subset of single-part MusicXML. The format may evolve and be extended in the future, however, Zeus pays little attention to this for two reasons:

1) With a new LMX version, a new Zeus model snapshot would likely need to be trained anyways.
2) LMX is designed to closely mirror MusicXML so any future versions should only consist of backwards-compatible additions and bugfixes. This means that old models should still output valid LMX, even though it would be just a subset of the full vocabulary.

For these reasons Zeus snapshots do not store the LMX version used to train them. This values must be "known" about the snapshot implicitly and thought about when using old models with possibly new versions of LMX.

In other words: **Although not explicit, keep LMX version in mind, though a mismatch likely should not cause any serious problems.**


## LMX/MusicXML semantics

The previous section discussed possible compatibility issues with LMX versioning, however the true devil is in semantics.

MusicXML was designed to encode complete music notation documents containing multiple pages of music. Zeus is built to recognise only individual staves or grandstaves. This limited viewing window of the score causes a lot of context to be unknown in the input image that Zeus receives (time signature, key signature, clef even). This has impact on the output MusicXML, which assumes this context is well known.

This section aims to define "best practises" for resolving these conflics, so that various existing Zeus model snapshots output mutually consistent MusicXML.

The "global context" concerning MusicXML mainly entails the `<attributes>` element and its children. In particular:

- `<divisions>`
- `<staves>`
- `<key>`
- `<time>`
- `<clef>`


### Technical semantics (divisions, staves)

**Ad `<divisions>`:**

MusicXML encodes duration of notes with both their type (quarter, half) and numeric value (number of beats in the divisions unit). This second numeric value is mainly used by audio-oriented consumers of MusicXML, so LMX ignores it, however, it should be present and be consistent so that the MusicXML document is valid.

If a note is a half note (two beats), and the initial `<attributes>` object in the given `<part>` specifies that `<divisions>` is 32 (32 units in one beat), then our half note has duration of 64 (twice one beat).

Slicing MusicXML naively and forgetting about the `<divisions>` element would make all the duration values unparsable. Luckily the [LMX python package](https://github.com/OMR-Research/lmx) provides MusicXML slicing methods (to slice out a given staff/grandstaff) that ensure this metadata is always present and valid.

LMX and **Zeus does not predict divisions** and durations, instead the LMX decoder automatically fills this data in based on the decoded note types.

**Ad `<staves>`:**

MusicXML represents instruments as parts (`<part>`) and instruments may have one or two staves. This is encoded in the `<staves>` element in the first measure. If this element is missing, the assumption is that the instrument only has 1 staff.

**Zeus does not predict the staves element** since a model is typically trained exclusively on solo-staff or grandstaff music. LMX decoder is what adds this element automatically. Moreover, it does that based on the clef elements present, so it tries to [crudely detect](https://github.com/OMR-Research/lmx/blob/8648514f5bfa08c594277b2d7821a156c75dd299/src/lmx/tokenization/Decoder.py#L102), whether the output of Zeus is actually solo-staff or grandstaff. But this detection may fail in edgecase for a combined model (one for both solo and grand staves), so might be re-implemented in the LMX package appropriately, possibly requesting the `<staves>` element to become an explicit LMX token.


### Musical semantics (key, time, clef)

All the `<key>`, `<time>` and `<clef>` elements behave similarly, in that in modern music notation, they should be repeated at the start of each system (with the exception of time signature), but in old music notation, they aren't - which is a problem.

Not knowing `<time>` is not a big problem, since it does not prevent decoding of the following music. So the `<time>` element leaves this conversation: **If `<time>` isn't written in the score on that staff, then it does not exist in MusicXML nor LMX.**

The other two - `<key>` and `<clef>` are needed, because they impact the pitch of notes (MusicXML encodes actual semantic pitch, while image encodes relative visual position). Missing a clef or key signature means we cannot correctly predict LMX tokens.

**If the staff begins with visible clef or key signature, they must be present in LMX and MusicXML.**

If they don't, then special care must be taken:

Because Zeus has not seen previous staves, it doesn't know which key to decode nota pitches in. Therefore a staff with missing clefs is assumed to have an invisible G-clef present. In MusicXML, this can be encoded with `print-object="no"` attribute on the `<clef>` and MuseScore 4 supports this attribute (clef is added and set as invisible).

However, the slicing methods of LMX know which clef preceeds the staff, so they work by adding in the proper clef and setting that to `print-object="no"`. Which means that for actual Zeus training, we need to replace this whichever clef with G-clef and transpose all following notes appropriately. Again, the LMX package provides helper methods for this invisible clef normalization.

The MusicXML and LMX data in training and evaluation datasets sent to Zeus (see [Zeus dataset format and pickling](zeus-dataset-format-and-pickling.md)) **must already be normliazed. The Zeus code does not perform any normalization during training.**

This also has implications on Zeus snapshots - a model trained without normalization will learn to guess the invisible clef and fail on normalized data evaluation. Keep this in mind!

Apart from clefs we also have key signatures. These are special in that when the image does not contain a key signature, it may mean the "no-accidentals" signature, or an unwritten actual key signature. In this second case, the LMX slicing methods insert the proper invisible `<key print-object="no">` signature and again, for training, the key signature must be normalized-away and notes transposed to pretend this staff actually begins with no key signature whatsoever.
