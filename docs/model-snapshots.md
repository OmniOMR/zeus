# Model snapshots

Zeus model may be stored to file system, which is called a snapshot. A snapshot is represented by a folder with the `.model` suffix. One snapshot of the `grand24` model is just under 30MB of disk space.

The snapshot folder contains the following files:

- `architecture_options.yaml` Architecture configuration values, see the `ArchitectureOptions` class (number of layers, dimensions, etc.).
- `token_map.txt` A map between string token names and token indices used in the model's embedding layer. Line number is the index and its content is the token name.
- `weights.h5` weights of the tensorflow model.

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
