# Visualizing data and predictions

Two commands render HTML pages you open in a browser. They answer different questions, and both are worth running before trusting a number.


## `zeus visualize-data` — what the model is actually fed

```bash
zeus visualize-data \
    --architecture solo26 \
    --train datasets/omniomr/samples.train.pickle \
    --augment "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2" \
    --batch-size 8 \
    --output out/training-data
```

Open `out/training-data/index.html`.

The images on that page are not the images on disk. They are the tensors the encoder receives, decoded back to PNG — after height normalization, after the width cap, and after every augmentation has had its turn. That is the whole point: augmentation settings are written as terse strings like `en3:0.2` and `de`, and the only reliable way to know what one does to your data is to look at it.

The LMX printed beside each image is likewise the sequence the model is told to produce, decoded back through the token map. If a token is missing from the map it shows as `<unk>`, which is worth noticing — see [tokenization](musicxml-lmx-and-tokenization.md).

`--architecture` is needed because the image height is an architecture setting, and it is what the pipeline normalizes to. No weights are involved and no model is trained; this shows the input pipeline, not a model.

Things this catches that a loss curve will not: augmentation strong enough to destroy the notation, a width cap squashing wide staves further than you expected, images that were cropped wrongly upstream, and samples whose LMX does not match their image at all.


## `zeus visualize-predictions` — where a model fails

```bash
zeus evaluate \
    --model-snapshot models/solo26.model \
    --dataset datasets/omniomr/samples.test.pickle \
    --output out/evaluation

zeus visualize-predictions \
    --dataset datasets/omniomr/samples.test.pickle \
    --predictions out/evaluation/predictions.lmx \
    --sample-count 100
```

The second command writes `out/evaluation/predictions.html` beside the predictions file it was given, with the images in a folder next to it.

Each sample shows its image, its symbol error rate, the gold LMX and the predicted LMX. **Samples are ordered by error, best first**, and that ordering is the value of the page: scrolling it walks from what the model reads perfectly to what defeats it, with the median error in the middle. A single average error rate says a model is imperfect; this says how.

It needs no model and no TensorFlow — the predictions have already been made — so it runs in about a second even on a large dataset.

The dataset must be the one that was evaluated. The nth prediction is matched to the nth sample by position, and a length mismatch is the only sign of the wrong dataset that anything can detect:

```
Given dataset has different number of samples than the precitions LMX file.
Did you provide the correct dataset?
```

A fixed random seed picks which samples to show, so re-running after retraining compares like with like rather than reshuffling the page.


## Rendering dataset images from MusicXML

A third command, `zeus render`, is not a visualization of a model but a way to produce images for a dataset that has MusicXML and no pictures — it engraves each sample through MuseScore. See [Zeus dataset format and pickling](zeus-dataset-format-and-pickling.md#rendering-musicxml-samples).
