# Training Zeus

The model can be trained via the `zeus train` CLI command. It can either:

- load an existing pre-trained snapshot and continue training,
- or create a completely new model and train from scratch.

Independently of that, it runs in one of two regimes. Either it trains for a fixed number of epochs given by `--epochs`, or it trains until the symbol error rate on a chosen dataset stops improving, with `--early-stopping`. The two are mutually exclusive and one of them is required.


## Training from scratch

To replicate the Camera GrandStaff model, use the following command:

```bash
zeus train \
    --experiment zeus-camera-grandstaff-replication \
    --architecture grand24 \
    --input-subdivisions Grandstaves \
    --train datasets/grandstaff/samples_distorted.train.pickle \
    --augment "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2" \
    --dev datasets/grandstaff/samples_distorted.dev.pickle \
    --test datasets/grandstaff/samples_distorted.test.pickle \
    --epochs 500 \
    --evaluation-from 10 \
    --evaluation-each 10 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --lr-decay cos \
    --quiet-tf
```

> **Note:** Start with a lower batch size, say 8 and go up and see if you even have enough RAM or GPU memory to run at 64.

To see the training progress, run `./tensorboard` and open `localhost:16006` in the browser.


## Fine-tuning an existing model

Fine-tuning on a new dataset, risks forgetting previously learned data:

> **Note:** Notice the learning rate is 10x smaller than in the example above and also the learning rate decay is disabled. Also note that the finetuning dataset is very small, so 20 epochs is really short. Adjust this according to your dataset size.

```bash
zeus train \
    --experiment zeus-olimpic-ft \
    --model-snapshot models/zeus-olimpic.model \
    --train datasets/authentic-piano/samples.finetune.pickle \
    --dev datasets/authentic-piano/samples.dev.pickle \
    --test datasets/authentic-piano/samples.test.pickle \
    --epochs 20 \
    --evaluation-from 1 \
    --evaluation-each 1 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --lr-decay none \
    --quiet-tf
```


## Early stopping

Picking the epoch count for a fine-tuning run is guesswork when the dataset is small: too few epochs leave the model unadapted, too many let it overfit, and the right number differs from one dataset to the next. Early stopping replaces the guess with a measurement. After every epoch, SER is computed on the dataset given to `--early-stopping`, and once it has not reached a new minimum for `--stopping-patience` epochs, training stops and the best epoch becomes the result:

```bash
zeus train \
    --experiment zeus-olimpic-ft \
    --model-snapshot models/zeus-olimpic.model \
    --train datasets/authentic-piano/samples.finetune.pickle \
    --dev datasets/authentic-piano/samples.dev.pickle \
    --test datasets/authentic-piano/samples.test.pickle \
    --early-stopping datasets/authentic-piano/samples.dev.pickle \
    --stopping-patience 5 \
    --evaluation-from 1 \
    --evaluation-each 5 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --quiet-tf
```

The early stopping dataset is a validation dataset in spirit, but it is its own argument. It may be one of the `--dev` datasets, as above, or a different one. The two are also measured on different schedules: `--dev` datasets are evaluated only on the `--evaluation-from` and `--evaluation-each` ticks, with every metric in `--metrics`, while the early stopping dataset is measured after every single epoch, and always by SER alone.

An epoch is a new minimum only when its SER is strictly lower than every epoch before it. The patience is how many epochs training continues past the latest minimum before it believes the SER has stopped falling, and it exists because the SER of a small dataset jitters from epoch to epoch — a single bad epoch is not yet a sign of overfitting. There is deliberately no default: choose it from how jittery the curves of similar runs have been. Too short a patience stops on noise; too long one only costs training time, since the result is the best epoch either way.

Early stopping requires a fixed learning rate. Cosine decay is a schedule over a known number of steps, which an early-stopped run does not have, and a decaying rate would flatten the SER curve for reasons other than the model having converged. `--lr-decay` therefore defaults to `none` under early stopping, and passing `--lr-decay cos` is an error.

The snapshots differ from the fixed-epoch regime in two ways. Any epoch might turn out to be the best one, so a snapshot is stored after every epoch — but once an epoch is further back than the patience, it can no longer become the result, and its snapshot is deleted unless it falls on an evaluation tick. What remains is exactly what the fixed-epoch regime would have kept, plus the last few epochs. And instead of a `final` snapshot, the run ends by restoring the weights of the best epoch and storing them as `ebest`, then evaluating the `--dev` and `--test` datasets on it.

The SER of every epoch so far, and which of them is the best, is written to `early_stopping.yaml` in the logdir after each epoch, and tensorboard plots it as `early_stopping_SER`. A run killed partway therefore still says which of its remaining snapshots to use.


## Command arguments

The `--experiment` argument provides name of the current training run (experiment), for use in tensorboard and logs.

Then you specify the model to be used. Either load an existing model snapshot via the `--model-snapshot` path, or specify the architecture to use for a new model via the `--architecture` argument. See the previous two sections for example values.

A new model also needs `--input-subdivisions`, which says which [Musicorpus page subdivisions](model-snapshots.md#architecture-options-versus-model-options) it will be trained to read — `Staves` for a solo-staff model, `Grandstaves` for a piano model, `Systems` for a whole system, or several of them for a model trained on more than one. It is stored in the snapshot, and it is what a Musibot worker announces, so it decides which images the deployed model is sent. Fine-tuning inherits it from the loaded snapshot, and passing it anyway overrides it.

Then you specify the training dataset(s) via one or many `--train` arguments. If many, then they are are concatenated and shuffled into one big training dataset. This may simulate fine-tuning with replay when the old training dataset is also included in the fine-tuning training process. Provide multiple values to a single `--train` option, not multiple `--train` options (that would cause only the last one to be loaded).

The training data may be augmented using the `--augment` argument, which then defines a set of filters and their parameters. For each sample, each filter is randomly applied with 50:50 change and a value (e.g. rotation amount) is uniformly sampled from a range specified as an argument of the filter. See `construct_tf_dataset.py` for more details.

For evaluation (continuous and terminal) specify the validation and evaluation datasets via `--dev` (validation) and `--test` (evaluation). Again, multiple values may be provided to have multiple validation/evaluation numbers. Validation runs on the evaluation ticks, controlled by `--evaluation-from` and `--evaluation-each` (delay start, then evaluate every N epochs), and again when training finishes. Evaluation on the `--test` datasets runs only when training finishes.

The length of training is given either by `--epochs`, a fixed number of epochs, or by `--early-stopping` together with `--stopping-patience`, which train until SER on the given dataset stops improving — see [Early stopping](#early-stopping). With a fixed epoch count, model weights are saved after each evaluated epoch, so you can still observe the validation curves and pick the appropriate snapshot afterwards. If unsure, set the epoch count higher than you expect, you can always kill the process and use whichever snapshot is the latest.

Which metrics the evaluations report is chosen with `--metrics`, a comma-separated list such as `SER,SERpitchonly`. It defaults to `SER` alone, and each metric becomes its own tensorboard curve named after the dataset and the metric. See [Evaluation metrics](evaluation-metrics.md).

Batch size is specified via `--batch-size`, use the highest that fits into GPU memory.

Learning rate is specified via `--learning-rate`, which may have a cosine decay if `--lr-decay cos` is used, otherwise the learning rate is constant. The decay defaults to `cos` with `--epochs` and to `none` with `--early-stopping`, which permits no other.

The `--quiet-tf` argument disabled Tensorflow warnings and verbose logs. You can set this flag when training regularly, however, if just starting the model for the first time on an unknown GPU, do NOT use this flag to see from the logs, whether Tensorflow actually uses the GPU provided or not.


## Tips

- Prefix the command with `time` to have the total runtime printed in the standard output.
- Start with a lower batch size, say 8 and go up and see if you even have enough RAM or GPU memory to run at 64.
- Remove `--quiet-tf` when testing a new GPU to see if it's acutally being used.
- Training time is specified with epochs, however, the total training steps then depend on dataset size (and batch size if that's not fixed). When combining multiple datasets and comparing results, adjust epochs to match the training time.
- When fine-tuning, lower learning rate and disable its decay.
- When fine-tuning on a small dataset, prefer `--early-stopping` over guessing `--epochs`.


## Output files

When you run training, it creates a folder `logs/{experiment-name}-{timestamp}` wich contains all the logs and results from the training process.

- `train/` Tensorboard events are logged into this folder
- `snapshots/` Contains one model snapshot for each evaluated epoch, named like `e5.model`
    - `final.model` the weights after the last epoch, with `--epochs` (the same as the last `e{N}.model`)
    - `ebest.model` the weights of the best epoch, with `--early-stopping`, which also leaves the snapshots of the last few epochs in place
- `evaluation/` Contains evaluation results for all evaluated epochs and the final epoch.
    - `e5-{eval_dataset_name}.yaml` evaluation metrics for a dataset at epoch 5
    - `e5-{eval_dataset_name}.lmx` evaluation predictions for a dataset at epoch 5 (one sample on one line)
    - `ebest-{eval_dataset_name}.yaml` and `.lmx` the same for the best epoch, with `--early-stopping`
- `early_stopping.yaml` With `--early-stopping`, the SER of every epoch and which one is the best so far.
- `training_options.yaml` The training options used for the training process.


## Publishing a snapshot

When you want to publish a model snapshot, include a `LICENSE` file. Say, CC BY-SA with [this content](https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt).

Then append a date to the model name, say converting `my-experiment.model` into `my-experiment-2026-07-10.model` and make sure it corresponds to the Musibot name and version in the `model_options.yaml` file to avoid confusion during deployment (adjust one or the other to match, keep the musibot version more detailed, with epoch number and such).

Finally, tar and zip the folder into a single file:

```bash
tar -czvf my-experiment-2026-07-10.model.tar.gz my-experiment-2026-07-10.model
```

Then attach the `.tar.gz` file to a GitHub release to be publicly available for download (or publish through any other channel).
