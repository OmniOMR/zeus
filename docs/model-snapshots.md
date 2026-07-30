# Model snapshots

Zeus model may be stored to file system, which is called a snapshot. A snapshot is represented by a folder with the `.model` suffix. One snapshot of the `grand24` model is just under 30MB of disk space.

The snapshot folder contains the following files:

- `architecture_options.yaml` Architecture configuration values, see the `ArchitectureOptions` class (number of layers, dimensions, etc.).
- `model_options.yaml` What the model reads and how it identifies itself, see the `ModelOptions` class and [the section below](#architecture-options-versus-model-options).
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

```yaml
input_subdivisions:
- Staves
musibot_model_name: solo26-omniomr
musibot_model_version: 2026-07-28-143052-e50
```

**`input_subdivisions`** says which [Musicorpus page subdivisions](https://github.com/OmniOMR/musicorpus/blob/main/docs/musicorpus-specification/musicorpus-specification.md) the model can read: `Staves` for a solo-staff model, `Grandstaves` for a piano model, `Systems` for a model reading a whole system at once. It is a set, so a model trained on both staves and grandstaves lists both, and the order it is written in carries no meaning.

**`musibot_model_name` and `musibot_model_version`** are the identity the snapshot announces when it is [served under Musibot](musibot-model.md). Together they are what a *Pipeline* pins, and Musibot treats both as opaque strings it never parses. Both may be absent, in which case they [fall back](#snapshots-from-before-this-file-existed).

They are prefixed because they exist for a consumer outside Zeus, and because `architecture_options.yaml` already has a `name` — holding `grand24` or `solo26` — one file away and meaning something else entirely.

The test for where a new setting belongs is the paragraph above the YAML. If two models that differ in it can still load each other's weights, it is a model option. A solo-staff model and a grandstaff model can share an architecture *exactly* — same height, same layers, same everything — and differ only in what you may hand them, which is precisely why that is not an architecture option.


### Why all three live in the snapshot rather than on the command line

Because each is a contract, not a note, and each fails silently when it is wrong.

**A wrong `input_subdivisions` transcribes nonsense confidently.** `zeus musibot` announces it as the input half of the model's [signature](https://github.com/OmniOMR/musibot/blob/main/docs/discovery.md), and Musibot routes work by that signature. A grandstaff model told that it reads staves does not crash and does not report a failure: it is handed staff crops, transcribes them, and returns music that was never on the page. Nothing in Musibot can detect it — the input is a perfectly valid image and the output is perfectly valid MusicXML. The error surfaces, if at all, as a quality complaint weeks later.

**A wrong name or version merges two models into one.** Musibot's registry keys a model by name and version, and treats a repeated pair as one model scaled horizontally. Two machines serving *different* snapshots that announce the same identity therefore collapse into a single registry entry, and executions land on whichever answers first — so a pipeline pinned to a model gets, at random, a different one.

A value that can be got wrong silently should not be restated at deployment time. Each is decided once, by whoever trained the model and knows the answer, and then travels with the weights it describes.

`input_subdivisions` is deliberately *not* derived from the architecture name either, tempting as that is — `grand24` and `solo26` do encode the intent, but by convention rather than by contract, and nothing stops a solo-staff model being trained on the `grand24` architecture.


### Snapshots from before this file existed

They simply do not have it, and every field falls back:

| Field | Falls back to |
| --- | --- |
| `input_subdivisions` | `Grandstaves`, which every Zeus model to date was bar one |
| `musibot_model_name` | `zeus` |
| `musibot_model_version` | the snapshot folder's own name, without the `.model` suffix |

The version fallback is more useful than it sounds, because a published snapshot folder is already named distinctively: `zeus-olimpic-1.0-2024-02-12.model` announces `zeus-olimpic-1.0-2024-02-12`, which is unique and readable. Two such snapshots cannot collide unless they are named the same.

If a fallback is wrong for a snapshot you have, you do not have to retrain it. Write the file into the snapshot folder by hand:

```bash
cat > models/my-solo-model.model/model_options.yaml <<'YAML'
input_subdivisions: [Staves]
musibot_model_name: zeus-solo
musibot_model_version: "2024-07-10"
YAML
```


### Setting it when training

`input_subdivisions` is the one thing a new model has nothing to inherit, so `zeus train` requires it:

```bash
zeus train --experiment solo26-omniomr --architecture solo26 --input-subdivisions Staves ...
```

Fine-tuning inherits it from the snapshot being loaded, and passing the argument anyway overrides it — which is what you want when a fine-tuning run deliberately changes what the model reads.

The identity needs no argument of its own. The **name is the `--experiment`**, so a fine-tuning run produces a differently-named model rather than a new version of its parent — which is right, since it is a different model. The **version is when the run started**, with the snapshot's own name appended:

```
logs/solo26-omniomr-260730_143052/snapshots/e50.model
    → musibot_model_version: 2026-07-30-143052-e50
```

That suffix is not decoration. `zeus train` stores a snapshot after every evaluated epoch, and picking the best one off the validation curve is the [documented workflow](training-zeus.md) — so every epoch of a run shares a start time, and without the suffix deploying `e40` and `e50` would announce the same identity and Musibot would merge them.

A fine-tuning run regenerates both from its own `--experiment` and start time rather than inheriting the parent's, for the same reason.
