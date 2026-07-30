# Model snapshots

Zeus model may be stored to file system, which is called a snapshot. A snapshot is represented by a folder with the `.model` suffix. One snapshot of the `grand24` model is just under 30MB of disk space.

The snapshot folder contains the following files:

- `architecture_options.yaml` Architecture configuration values, see the `ArchitectureOptions` class (number of layers, dimensions, etc.).
- `model_options.yaml` What the model reads, see the `ModelOptions` class and [the section below](#architecture-options-versus-model-options).
- `token_map.txt` A map between string token names and token indices used in the model's embedding layer. Line number is the index and its content is the token name.
- `weights.h5` weights of the tensorflow model.
- `LICENSE` optional license file, must be added manually when publishing a snapshot.

There is also a legacy representation from 2024, however the loading function will detect it and load appropriately.

A model can be stored like this:

```py
zeus = Zeus(...)

zeus.store(Path("models/my-model.model"))
```

And it can be loaded like this:

```py
zeus = Zeus.load(Path("models/my-model.model"))
```


## Architecture options versus model options

Two of those files hold settings, and which setting belongs in which is worth being precise about, because the answer is not "whichever seems related".

**`architecture_options.yaml` defines the computation graph.** Every value in it — the image height, the CNN stages, the RNN dimensions, the timestep width — changes the shape of a tensor somewhere. Two models with equal architecture options have *identical* graphs, and that is the property the file exists to express: it is what lets `Zeus.load` rebuild the network before it has any weights to put in it. The named architectures, `grand24` and `solo26`, are presets of exactly these values.

**`model_options.yaml` holds facts about the trained model that its graph cannot express.** They change nothing about the network and everything about how it may be used.

Today it holds one thing:

```yaml
input_subdivisions:
- Staves
```

which says which [Musicorpus page subdivisions](https://github.com/OmniOMR/musicorpus/blob/main/docs/musicorpus-specification/musicorpus-specification.md) the model can read: `Staves` for a solo-staff model, `Grandstaves` for a piano model, `Systems` for a model reading a whole system at once. It is a set, so a model trained on both staves and grandstaves lists both, and the order it is written in carries no meaning.

The test for where a new setting belongs is that first paragraph. If two models that differ in it can still load each other's weights, it is a model option. A solo-staff model and a grandstaff model can share an architecture *exactly* — same height, same layers, same everything — and differ only in what you may hand them, which is precisely why this is not an architecture option.


### Why it lives in the snapshot rather than on the command line

Because it is a contract, not a note. `zeus musibot` announces it to Musibot as the input half of the model's [signature](https://github.com/OmniOMR/musibot/blob/main/docs/discovery.md), and Musibot routes work by that signature.

So consider what a wrong value does. A grandstaff model told that it reads staves does not crash, and does not report a failure: it is handed staff crops, transcribes them confidently, and returns music that was never on the page. Nothing in Musibot can detect this — the file it was given is a perfectly valid image, and the MusicXML that comes back is perfectly valid MusicXML. The error surfaces, if at all, as a quality complaint weeks later.

A value that can be got wrong silently should not be restated at deployment time. It is decided once, by whoever trained the model and knows the answer, and then travels with the weights it describes.

It is deliberately *not* derived from the architecture name, tempting as that is — `grand24` and `solo26` do encode the intent, but by convention rather than by contract, and nothing stops a solo-staff model being trained on the `grand24` architecture.


### Snapshots from before this file existed

They simply do not have it, and are read as `Grandstaves`, which every Zeus model to date was bar one.

If you have such a snapshot and that is wrong, you do not have to retrain it. Write the file into the snapshot folder by hand:

```bash
echo 'input_subdivisions: [Staves]' > models/my-solo-model.model/model_options.yaml
```


### Setting it when training

A new model has nothing to inherit this from, so `zeus train` requires it:

```bash
zeus train --architecture solo26 --input-subdivisions Staves ...
```

Fine-tuning inherits it from the snapshot being loaded, and passing the argument anyway overrides it — which is what you want when a fine-tuning run deliberately changes what the model reads.
