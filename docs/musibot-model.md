# Running Zeus as a Musibot model

[Musibot](https://github.com/OmniOMR/musibot) is where trained OMR models are deployed and used in production. It runs each *Model* as an isolated subprocess: a *Worker Head* starts it, hands it two file descriptors, and drives it with JSON lines, while images and transcriptions travel through a directory.

`zeus musibot` is Zeus behind that contract.

```bash
zeus musibot --model-snapshot models/solo26.model
```

It is never started by hand — a worker head starts it and supplies the descriptors. Run without them it says so and exits rather than failing with a `KeyError`.


## What it announces

On startup Zeus loads the snapshot and then announces itself. Loading first is deliberate: a worker head consumes no work and announces nothing until its model has spoken, so the seconds spent loading weights are seconds during which this worker is simply not offered any.

```json
{
  "type": "ready",
  "ipc_version": 1,
  "model": {
    "name": "zeus-solo-omniomr",
    "version": "2026-07-28-143052-e50",
    "signature": {
      "input": ["Staves/{staff}/image.jpg"],
      "output": [
        "Staves/{staff}/transcription.musicxml",
        "Staves/{staff}/transcription.lmx"
      ]
    },
    "supports_batching": true
  }
}
```

Everything in there comes from the snapshot rather than from the command line, because the name and version are what a *Pipeline* pins and what Musibot's registry keys a model by. Two workers announcing the same name and version are taken to be one model scaled horizontally — so if this were a deployment-time flag, two machines serving *different* snapshots could collide into one registry entry and executions would land on whichever answered first. See [Model snapshots](model-snapshots.md) for where the values come from and how to change them.


### Reading the signature

`{staff}` is a [slot](https://github.com/OmniOMR/musibot/blob/main/docs/signatures.md): it stands for one subdivision instance, whatever it happens to be called. So the signature is true of every page rather than of one — `Staves/7/image.jpg` and `Staves/2-3/image.jpg` both fit it, and the Musicorpus Specification allows an instance to be named with any path-safe string.

Two things follow from how it is written.

**The same name on both sides** says the transcription lands beside the image it came from. Read staff 7, write staff 7.

**`{staff}` and not `{*staff}`** says one staff per execution rather than all of them at once, which is a claim about how Zeus works rather than a convenience. Zeus transcribes a staff without reference to any other, so the unit of work is one instance — and keeping that the same as the unit of reporting is what lets a batch of twelve staves report twelve outcomes, so that one unreadable staff fails only itself. A `{*staff}` model has a single indivisible outcome and could only fail all twelve or silently skip one. Batching is how they still go through the model together, filling one forward pass.

The practical consequence is that the `api` service now refuses a wrong input list before it reaches a worker, with a message naming the mismatch:

```
The input file 'image.jpg' is not named by the signature (Staves/{staff}/image.jpg)
'Staves/{staff}/image.jpg' names one file, but the input list has 2: …
```

That first one matters more than it looks. Zeus handed a whole page does not complain — it transcribes the page as though it were one staff and returns confident nonsense — and the [ImplicitPipeline](https://github.com/OmniOMR/musibot/blob/main/docs/domain-model.md) generated for a model is exactly its signature, so this is what stops a *User* running Zeus over a page image and believing the result.

A model that reads more than one subdivision announces every entry as optional:

```json
"input": ["Grandstaves/{grandstaff}/image.jpg?", "Staves/{staff}/image.jpg?"]
```

because every non-optional entry has to be matched, and two required ones would demand a staff *and* a grandstaff in the same execution. That is wider than the truth — it also admits an execution naming both, or neither — and Zeus refuses those itself with `Zeus reads exactly one image per execution`. Declaring the wider signature and reporting a plain error for what satisfies it but not the model is what Musibot prescribes for expectations a signature cannot express.


## What it reads and writes

One image per execution, and the transcription is written beside it:

```
7Kf2mP9xLwQ/
  Staves/
    1/
      image.jpg                  ← read
      transcription.musicxml     ← written
      transcription.lmx          ← written
```

`transcription.musicxml` is the [Musicorpus Specification](https://github.com/OmniOMR/musicorpus/blob/main/docs/musicorpus-specification/musicorpus-specification.md) file for a staff's transcription, and where the spec says it goes.

`transcription.lmx` is **not** a Musicorpus file type. It is the raw token sequence the model actually predicted, before it was turned into MusicXML, and it is written because that is what you want when a transcription looks wrong — the MusicXML is a derived artifact, and a decoding problem is invisible in it. Musicorpus is deliberately extensible in this way: what is not in the specification may be used as long as it is documented, which is what this paragraph is. Pass `--no-lmx` to write only the MusicXML.

Which subdivision Zeus accepts — `Staves`, `Grandstaves`, `Systems`, or several — is [declared by the snapshot](model-snapshots.md#architecture-options-versus-model-options). An execution naming a subdivision the model does not read is refused:

```
This model reads Grandstaves, but was given 'Staves/1/image.jpg'.
```

That check is the point of storing it. A grandstaff model handed staff crops does not crash — it transcribes them confidently and returns music that was never on the page — so the mismatch has to be caught deliberately or not at all.


## Batching

Zeus advertises `supports_batching`, so a worker head may send an `execute-batch` carrying several executions at once. They go through the model in a single forward pass, which is most of what makes inference fast.

The executions in a batch may come from different pipelines and different pages. Each is reported separately, and one failing does not fail the others: reading the image, writing the outputs and decoding the LMX all happen per execution, and only the forward pass is shared.

`--batch-size` bounds how many images enter one forward pass. It is not the same number as the batch a head sends; it is what happens to that batch here.


## What a failure looks like

Failures are reported per execution and reach the *Pipeline Execution* log, so they are written for a human to read. The ones you will actually see:

| Message | Means |
| --- | --- |
| `There is no file at 'Staves/9/image.jpg'.` | The head staged something else, or the pipeline named a staff that does not exist. |
| `This model reads Grandstaves, but was given 'Staves/1/image.jpg'.` | A pipeline is routing the wrong subdivision to this model. |
| `Could not decode the predicted LMX into MusicXML: …` | The model's output was not well-formed LMX. A prediction is not a promise; this is an ordinary outcome for a hard image. |
| `Zeus reads exactly one image per execution, but 2 input file(s) were given.` | Only reachable for a model reading several subdivisions, whose signature has to be wider than the truth. For a single-subdivision model the `api` service refuses this first. |

A model that dies fails its work and reports nothing useful, so `zeus musibot` reports and keeps serving instead: whatever goes wrong inside one execution is caught, reported as that execution's failure, and the loop continues. An execution the model somehow never reports is failed on its behalf rather than left to time out.


## Deployment: two virtual environments

Zeus needs **python 3.10** — TensorFlow 2.12 has no wheels for anything newer — while a Musibot worker head depends on `musibot-core` and needs **python 3.11 or newer**. The two cannot share an environment even in principle, which is precisely the case the [worker IPC](https://github.com/OmniOMR/musibot/blob/main/docs/worker-ipc.md) boundary exists for. Nothing crosses it but bytes.

So Zeus gets a venv of its own and the worker head is pointed at it by path:

```bash
# Zeus, on python 3.10
python3.10 -m venv /opt/zeus/.venv
/opt/zeus/.venv/bin/pip install 'zeus @ git+https://github.com/OmniOMR/zeus.git@v1.0.0'

# the worker head, on python 3.11+
python3.11 -m venv /opt/worker-head/.venv
/opt/worker-head/.venv/bin/pip install \
  'musibot-core @ git+https://github.com/OmniOMR/musibot.git@core/v0.1.0#subdirectory=components/core' \
  'musibot-worker-head @ git+https://github.com/OmniOMR/musibot.git@worker-head/v0.1.0#subdirectory=components/worker-head'

# and the head runs Zeus across the boundary
/opt/worker-head/.venv/bin/musibot-worker-head \
    --model-command "/opt/zeus/.venv/bin/zeus musibot --model-snapshot /opt/zeus/models/solo26.model --quiet-tf" \
    --rabbit-host rabbit.internal \
    --s3-endpoint-url http://minio.internal:9000
```

Model weights are the model repository's business, never Musibot's; see [Model snapshots](model-snapshots.md) for where Zeus keeps its.

> **Note:** passing file descriptors to a child is POSIX-only. Zeus can be developed anywhere, but it has to reach Linux (or WSL) before it can run under a worker head.
