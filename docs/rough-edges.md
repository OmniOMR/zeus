# Rough edges

Known limitations, in the same spirit as [Musibot's](https://github.com/OmniOMR/musibot/blob/main/docs/rough-edges.md): things that are missing, awkward, or right only by convention. Written down because a limitation that is documented is a decision, and one that is not is a trap.


## The signature cannot say what Zeus reads

A Musibot *Signature* is a flat list of file paths, and Zeus reads *any* staff of a page — however many there are, named however the dataset named them. So `zeus musibot` announces `Staves/1/image.jpg` as a placeholder standing in for all of them.

Nothing breaks: the announcement is not consulted when work arrives, and an execution naming `Staves/7/image.jpg` is served identically. What suffers is discoverability — a *Pipeline* author cannot learn from the listing that Zeus wants calling once per staff, and the *ImplicitPipeline* generated for Zeus offers the page-level `image.jpg`, which Zeus will transcribe as though the whole page were one staff.

Fixing it means growing the *Signature* format in `musibot-core`. The shape it would want is written down in [Running Zeus as a Musibot model](musibot-model.md#the-signature-cannot-say-what-zeus-actually-reads).


## A malformed prediction crashes the `lmx` decoder

An LMX string with tokens before its first `measure` makes the pinned `lmx` release raise `AttributeError` from inside its own error-reporting path — `Decoder.py` passes a message where a `Token` is expected. Models do emit such sequences, especially early in training.

Zeus does not propagate it: `lmx_to_musicxml` raises `LmxDecodingError` carrying the offending LMX, and callers fail that one sample. But the message a user sees still ends in `'str' object has no attribute 'terminal'`, which explains nothing. The fix is one line in [lmx](https://github.com/OMR-Research/lmx) and a bump of the pin in `pyproject.toml`.


## MusicXML-level evaluation does not exist

`zeus pickle --with-musicxml` stores each sample's MusicXML in the pickle, and `ZeusDatasetSample.musicxml` carries it into memory, but nothing reads it. It is there for TEDn — a tree edit distance over the MusicXML rather than over the token sequence — which is not implemented.

So evaluation today is symbol error rate on LMX, which is sensitive to how the notation was linearized. Two transcriptions that a musician would call equivalent can differ in LMX, and SER counts that as error. Until TEDn exists, treat SER as a training signal rather than as a claim about musical correctness.


## `zeus musicorpus` cannot re-crop or rescale

`--re-crop` and `--normalize-image-height` are accepted and then refused before any work starts. Both would be useful — the first for datasets whose crops are poor, the second for controlling resolution at conversion time rather than at training time — and neither is implemented. Sample images are copied through unchanged.


## Snapshots from before `model_options.yaml` guess

A snapshot without that file is read as a grandstaff model, which every Zeus model to date was bar one. The exception is a solo-staff model that will announce itself wrongly to Musibot until either retrained or corrected by hand. Writing the file into the folder is enough; see [Model snapshots](model-snapshots.md#snapshots-from-before-this-file-existed).


## Python 3.10, and no way out that is cheap

TensorFlow 2.12 has no wheels for python 3.11 or newer, and the model's weights are Keras 2 checkpoints. So Zeus is pinned to 3.10, which forces the two-virtual-environment arrangement when deploying under a Musibot worker head, and will eventually force a migration — either to a newer TensorFlow or off it. Nothing here is urgent while the model trains and runs, but the pin only gets more awkward with time.


## The width cap distorts wide staves

Images wider than `max_image_width` at the architecture's height are compressed horizontally rather than scaled down, so that the height stays exact. That is the right trade — vertical resolution is where pitch lives — but it does mean a very wide staff reaches the model with its note spacing squeezed, and how much that costs has not been measured. The cap can be lifted with `max_image_width=None`, at the price of larger tensors.


## `zeus predict` writes beside its input

By default each transcription lands next to the image it came from, which is convenient for a folder of staves and surprising if the input folder is meant to be read-only. `--output-dir` redirects it. There is no way to write to standard output.
