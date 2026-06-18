# Zeus dataset format and pickling

The Zeus model requires a specific dataset format for training, evaluation and inference. This format is called the *Zeus dataset format*. This format is optimized for feeding data into the model.


## Zeus dataset format

A Zeus dataset is a folder with the following structure:

```
my-dataset/ 📚
│
│   (🪓 samples files define which samples belong to which split)
├── samples.train.txt 🪓
├── samples.dev.txt 🪓
├── samples.test.txt 🪓
├── samples.{split}.txt 🪓
│
├── README.md       (optional root files)
├── LICENSE         (optional root files)
│
│   (📜 list of samples in any hierarchy)
└── samples/
    ├── any-subfolders/
    │   │
    │   │   (📜 this is "samples/any-subfolders/my-sample-1" sample)
    │   ├── my-sample-1.jpg 🖼️          (may also be png)
    │   ├── my-sample-1.lmx 🎼
    │   ├── my-sample-1.musicxml 📄     (optional)
    │   │
    │   ├── my-sample-2.jpg 🖼️
    │   ├── my-sample-2.lmx 🎼
    │   ├── my-sample-2.musicxml 📄
    │   │
    │   └── ...
    │
    └── other-subfolders/
        └── ...
```

In the dataset root folder, there are `samples.{split}.txt` files. These are called samples files (see `SamplesFile` in Python). The file simply lists one sample per line, thereby specifying which split contains which samples:

```txt
samples/any-subfolder/my-sample-1
samples/any-subfolder/my-sample-2
samples/any-subfolder/my-sample-3
samples/any-subfolder/my-sample-4
samples/other-subfolder/my-other-sample-1
samples/other-subfolder/my-other-sample-2
samples/other-subfolder/my-other-sample-3
```

Each line is called *sample name* and it's a POSIX path relative to the sample file to the relevant data files (jpg, png, lmx, musicxml), just without the file extension suffix.

Data files for samples should be nested inside the `samples` folder, although this is only a recommendation. The internal structure of this folder is up to the dataset creator and should be such that it eliminates excessive number of files per directory to not freeze up file explorers when looking at the data manually.

One dataset *split* (train, dev, test, any custom) is represented by the path to the corresponding *samples file* and when loaded in memory in Python, it's represented by the `ZeusDataset` class.


### Image data

Images should be provided in JPEG or PNG format (each sample can be different), whichever better suits the data (binarized or born-digital B/W data is better with PNG, scanned with JPEG). The resolution can be arbitrarily large, images will get rescaled before being fed into the model, however, to reduce disk usage and speed up decoding, you can scale them down here already.

The image should be a crop of a single staff/grandstaff (based on your model), generally wider than taller.

Image files are mandatory (the model needs input).


### LMX data

The `my-sample.lmx` files are text files containing LMX ([Linearized MusicXML](https://github.com/OMR-Research/lmx)) tokens, encoded as space-separated strings. The file has only one line (line breaks should not appear in LMX).

This is an example content of the LMX file:

```txt
measure key:fifths:5 time beats:4 beat-type:4 clef:G2 staff:1 ...
```

LMX files are mandatory for training and evaluation (the model needs expected output).

For more information about the semantics of LMX tokens, conversion from MusicXML and other edgecases, see the [MusicXML, LMX and tokenization](musicxml-lmx-and-tokenization.md) documentation page.


### MusicXML data

Optionally, samples may contain `.musicxml` data files. These MusicXML files should contain the exact MusicXML from which the LMX representation was generated.

These files are optional, but for the whole dataset. Either all samples should have it, or all files should miss it.

Its only purpose is to allow for MusicXML-to-MusicXML evaluation, such as tree edit distance, musicdiff, or music tree notation. It is not used for model training nor inference.
