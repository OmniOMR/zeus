# Converting MusiCorpus datasets to Zeus format

[MusiCorpus](https://github.com/OmniOMR/musicorpus) is a standard for structuring OMR datasets. This zeus package provides a CLI command for converting MusiCorpus datasets into the Zeus format, so that they can be used for Zeus training.

This is the basic command to use:

```bash
zeus musicorpus \
    --input ~/musicorpus/CVC.Dolores \
    --take_staves \
    --output datasets/dolores
```

The conversion requires these files to be present in the input MusiCorpus dataset:

- `splits.json 🪓` in the dataset root is used to generate Zeus `samples.{split}.txt` files.
- `image.jpg 🖼️` at the staff or grandstaff level is required, optionally page-level when doing re-cropping
- `transcription.musicxml 📄` at the staff or grandstaff level is required
- `layout.json 📏` only needed when doing re-cropping

The output is a new folder containing the Zeus representation of the extracted data, e.g. `my-dataset`. It contains the `samples.{split}.txt` files and the `samples` folder with images, MusicXML and LMX. The command performs necessary semantic corrections of MusicXML as described in [MusicXML, LMX and tokenization](musicxml-lmx-and-tokenization.md) documentation page. The output dataset also getes a `samples.all.txt` file with all samples extracted from the dataset, if splits are not to be used.

The command must be instructed to take either staves or grandstaves as samples, or possible both, by usigin these two toggles:

```
--take_staves
--take_grandstaves
```

The command also supports these additional flags:

**`--force`** Overwrite the output folder, if it exists.

**`--re_crop`** Instead of using staff/grandstaff level `image.jpg` files, the command loads the `layout.json` file and performs crops manually, with special attention to the distribution of crop region position to the staff position (to make the trained model resilient to imprecise crops). Do this only with training data not evaluation so that evaluation results are comparable across models.

**`--normalize_image_height`** Pre-compress sample images to a specific height to reduce Zeus dataset size. The argument is the number of pixels.
