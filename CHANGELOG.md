# Changelog for Zeus

Released as `vX.Y.Z` git tags — see [Versioning and releases](docs/versioning-and-releases.md).

Entries are written for whoever installs Zeus, so they describe behaviour and contracts rather than commits.

This file records the package version. A trained model carries its own version, which a *Pipeline* pins and which does not move when Zeus is released; see [Model snapshots](docs/model-snapshots.md).


## Unreleased

First release in preparation. Everything below is new relative to the code inherited from the [ICDAR 2024 paper](https://github.com/ufal/olimpic-icdar24).


### Added

- **`zeus musibot`** — runs Zeus as a [Musibot](https://github.com/OmniOMR/musibot) *Model*, speaking the worker IPC contract (`ipc_version` 1) over the two file descriptors a *Worker Head* provides. It announces a slotted signature — `Staves/{staff}/image.jpg` — so the declaration is true of every page, the transcription is bound to the staff it came from, and the `api` service refuses a page image or a dozen staves before they reach a worker. Batching is supported, so one `execute-batch` fills a single forward pass. Transcriptions are written beside the image they came from, as `transcription.musicxml` and `transcription.lmx`. See [Running Zeus as a Musibot model](docs/musibot-model.md).
- **`zeus predict`** — reads music notation off staff images and writes MusicXML, or the raw LMX tokens with `--lmx`, or only those with `--lmx --no-musicxml`. `Zeus.predict` previously raised `NotImplementedError`, and every path that ran the model required gold LMX for every sample.
- **`model_options.yaml` in a snapshot** — declares which Musicorpus page subdivisions the model reads (`Staves`, `Grandstaves`, `Systems`), and the name and version it announces to Musibot. Snapshots written before this file existed read as `Grandstaves`, and can be corrected by writing the file into the folder by hand rather than retraining. See [Model snapshots](docs/model-snapshots.md).
- **A public python API** — `from zeus import Zeus, InferenceOptions, …`. The names are resolved lazily, so `import zeus` costs nothing and does not import TensorFlow. See [Python API](docs/python-api.md).
- **`--input-subdivisions` on `zeus train`**, required when training a new model, inherited from the snapshot when fine-tuning.
- **A named set of evaluation metrics** in `zeus.evaluation`, selectable with `--metrics` on both `zeus train` and `zeus evaluate`, or with `metrics=` in python. `SER` remains the default and the only one computed unless others are asked for. `SERpitchonly` is new — it counts pitch tokens alone, separating whether the model read the right notes from whether it got their durations right. See [Evaluation metrics](docs/evaluation-metrics.md).
- **`--sample-count` on `zeus visualize-predictions`**, which was fixed at 100.
- Documentation: [Python API](docs/python-api.md), [Running Zeus as a Musibot model](docs/musibot-model.md), [Versioning and releases](docs/versioning-and-releases.md), [Repository layout](docs/repository-layout.md), [Visualizing data and predictions](docs/visualizing-data-and-predictions.md), [Rough edges](docs/rough-edges.md), and an account of what belongs in `architecture_options.yaml` versus `model_options.yaml`.


### Changed

- **The version is derived from git tags** rather than written into `pyproject.toml`, so every commit installs as a distinct version over a `pip install` from a git URL. Previously a reinstall from a newer commit did nothing, silently.
- **Commands and flags are kebab-case** throughout: `visualize_data` → `visualize-data`, `--batch_size` → `--batch-size`, and so on. Two are renamed for meaning: `--model` → `--model-snapshot`, and `--new_model` → `--architecture`, which is what it always took.
- **`zeus visualize-predictions` no longer takes `--architecture`** and no longer loads TensorFlow, since the predictions have already been made. `--predictions` is now required; its default named a path that could not exist.
- Modules are named in snake_case, and the package is importable as documented rather than by internal path.
- `zeus musicorpus` refuses `--re-crop` and `--normalize-image-height` before doing any work, rather than exiting with status 0 partway through.
- **`SERnotuplets` is no longer computed on every evaluation.** It answers a question specific to corpora full of implicit tuplets, and reporting it always made every run carry a number most of them had no use for. Ask for it with `--metrics` when it is the one you want.


### Removed

- **MusicXML no longer travels in dataset pickles.** `zeus pickle --with-musicxml` and `ZeusDatasetSample.musicxml` are gone; nothing read the field, and evaluating at the MusicXML level needs decoding, is expensive, and belongs to a benchmarking rig run after training rather than to the repository that trains. Zeus is LMX-first. The `.musicxml` files in a dataset folder stay — `zeus render` engraves sample images from them — and `zeus predict` still decodes MusicXML as a convenience. Pickles written before this still load; the stale field is ignored.


### Fixed

- **Images wider than `max_image_width` broke training, evaluation and prediction.** The resize preserved the aspect ratio inside a box, so any staff wider than the cap came out shorter than the architecture's height and the encoder failed with an unreadable reshape error. Two thirds of a typical solo-staff dataset is that wide. The height is now exact and only the width is compressed, since vertical resolution is where pitch lives.
- **`Zeus.store` followed by `Zeus.load` was not a round trip.** Storing a model that had not yet decoded anything wrote a weights file containing the encoder alone, and loading it failed later with a layer count mismatch.
- **A malformed prediction no longer crashes the process.** `lmx_to_musicxml` raises `LmxDecodingError` carrying the offending LMX, and callers fail that one sample rather than the run — a model's output is a prediction, not a promise.
- **`<unk>` was invisible in both visualizations.** It is a real token, and written unescaped into HTML a browser reads it as an unknown tag and shows nothing, hiding the token most worth noticing.
- `ser_metric` was annotated as returning a float while returning a dict of metrics, which every caller indexes.
- A bare `except:` around the `Levenshtein` import swallowed `KeyboardInterrupt` and silently fell back to the pure-python implementation, some hundred times slower over a full evaluation.
- The entry point was declared as a `gui-script`, which builds a no-console executable on Windows.
- `requires-python` was `~=3.10`, which permits 3.11 and 3.12 — where TensorFlow 2.12 has no wheels, so the installation could not work.
- The training documentation instructed passing an architecture named `grand26`, which does not exist.


### Notes

- Requires **python 3.10**, and cannot run on anything newer while TensorFlow 2.12 is the pinned version. Through the Musibot worker IPC boundary this is a constraint on deployment rather than on the rest of the system; see [the deployment section](docs/musibot-model.md#deployment-two-virtual-environments).
- Dataset pickles written by earlier versions still load: the modules they name were renamed, and the old paths are redirected.
