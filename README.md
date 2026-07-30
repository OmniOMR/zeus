# Zeus

<div align="center">
    <br/>
    <img src="docs/zeus-hero-image.png" width="500px"/>
    <br/>
    <br/>
</div>

Zeus is a deep learning model for reading staves and grandstaves of music notation. This repository is structured as a python package that lets you train Zeus, save and load it from checkpoints and use it for inference.


## Usage

You have two ways how to use Zeus for your project:

1. Use just the CLI - install Python 3.10, setup a venv and install this package into it
2. Use the Python API - your project already runs Python 3.10, simply install this package

In other words, if you can't afford to have your project on Python 3.10, you must use Zeus via its CLI. Otherwise you can also use its [Python API](docs/python-api.md), whose whole surface is importable from the package root:

```py
from zeus import Zeus, InferenceOptions
```

This is the command to install Zeus from this GitHub repository at the latest commit:

```bash
pip install 'zeus @ git+https://github.com/OmniOMR/zeus.git@main'
```

Replace `main` with a release tag such as `v1.0.0` to pin a version. Every commit builds as a distinct version, so a plain `pip install` of a newer commit replaces what is there — see [Versioning and releases](docs/versioning-and-releases.md).

Learn more about [VCS support](https://pip.pypa.io/en/stable/topics/vcs-support/) of `pip`.


## Existing snapshots

This is a list of known Zeus model snapshots available for download:

- Latest solo-staff 2026 experiments (TODO license)
    - `[2024-07-10]` **`zod-bw-auth-2024-07-10.model`** (TODO) trained on Dolores
    - `[2024-07-10]` **`zod-bw-auth-ft-2024-07-10.model`** (TODO) trained on Dolores, finetuned on OmniOMR
- Original 2024 grandstaff B/W models ([code](https://github.com/ufal/olimpic-icdar24/releases), [paper](https://doi.org/10.1007/978-3-031-70552-6_4)) (CC BY-SA licenses)
    - `[2024-02-12]` **`zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`** ([download](https://github.com/ufal/olimpic-icdar24/releases/download/grandstaff-models/zeus-camera-grandstaff-lmx-1.0-2024-02-12.model.tar.gz), [url](https://github.com/ufal/olimpic-icdar24/releases/tag/grandstaff-models)) trained on Camera GrandStaff
    - `[2024-02-12]` **`zeus-grandstaff-lmx-1.0-2024-02-12.model`** ([download](https://github.com/ufal/olimpic-icdar24/releases/download/grandstaff-models/zeus-grandstaff-lmx-1.0-2024-02-12.model.tar.gz), [url](https://github.com/ufal/olimpic-icdar24/releases/tag/grandstaff-models)) trained on GrandStaff
    - `[2024-02-12]` **`zeus-olimpic-1.0-2024-02-12.model`** ([download](https://github.com/ufal/olimpic-icdar24/releases/download/zeus-release/zeus-olimpic-1.0-2024-02-12.model.tar.gz), [url](https://github.com/ufal/olimpic-icdar24/releases/tag/zeus-release)) trained on OLiMPiC


## CLI

The package exposes the `zeus` CLI command, which should be used to work with the model:

- `zeus` **`train`** [`--help`](docs/training-zeus.md): Trains a Zeus model, both new or loaded.
- `zeus` **`evaluate`** [`--help`](docs/using-the-cli.md#evaluating-a-snapshot-against-a-dataset): Performs symbol error rate evaluation on LMX against the given dataset split.
- `zeus` **`predict`** [`--help`](docs/using-the-cli.md#transcribing-images): Reads music notation off staff images and writes MusicXML transcriptions.
- `zeus` **`musibot`** [`--help`](docs/musibot-model.md): Runs Zeus as a [Musibot](https://github.com/OmniOMR/musibot) model, driven by a worker head over pipes.
- `zeus` **`visualize-data`** [`--help`](docs/visualizing-data-and-predictions.md): Renders training data, as the model sees it, into a browsable HTML page.
- `zeus` **`visualize-predictions`** [`--help`](docs/visualizing-data-and-predictions.md): Renders the output of `evaluate` beside its gold data, ordered by symbol error rate.
- `zeus` **`pickle`** [`--help`](docs/zeus-dataset-format-and-pickling.md#pickling): Bundles one Zeus dataset split into a pickle file for faster loading on the compute cluster.
- `zeus` **`musicorpus`** [`--help`](docs/converting-musicorpus-datasets-to-zeus-format.md): Converts a [MusiCorpus](https://github.com/OmniOMR/musicorpus) dataset to Zeus dataset.
- `zeus` **`render`** [`--help`](docs/zeus-dataset-format-and-pickling.md#rendering-musicxml-samples): Renders one Zeus dataset split via MuseScore to B/W images. Uses MusicXML files as input.


## Documentation

Using the model:

- [Using the CLI](docs/using-the-cli.md)
- [Python API](docs/python-api.md)
- [Model snapshots](docs/model-snapshots.md)
- [Running Zeus as a Musibot model](docs/musibot-model.md)

Training the model:

- [Training Zeus](docs/training-zeus.md)
- [Augmentations](docs/augmentations.md)
- [Visualizing data and predictions](docs/visualizing-data-and-predictions.md)
- [Training on the UFAL LRC cluster](docs/training-on-lrc.md)

Data:

- [MusicXML, LMX and tokenization](docs/musicxml-lmx-and-tokenization.md)
- [Zeus dataset format and pickling](docs/zeus-dataset-format-and-pickling.md)
- [Converting MusiCorpus datasets to Zeus format](docs/converting-musicorpus-datasets-to-zeus-format.md)

Design and technical:

- [Development](docs/development.md)
- [Model architecture](docs/model-architecture.md)
- [Repository layout](docs/repository-layout.md)
- [Versioning and releases](docs/versioning-and-releases.md)
- [Rough edges](docs/rough-edges.md)


## How to cite

The model is derived from its original design from 2024. If you use this repository, please cite the following paper:

Jiří Mayer, Milan Straka, Jan Hajič jr., Pavel Pecina. Practical End-to-End Optical Music Recognition for Pianoform Music. *18th International Conference on Document Analysis and Recognition, ICDAR 2024.* Athens, Greece, August 30 - September 4, pp. 55-73, 2024.<br/>
**DOI:** https://doi.org/10.1007/978-3-031-70552-6_4<br/>
**GitHub:** https://github.com/ufal/olimpic-icdar24
