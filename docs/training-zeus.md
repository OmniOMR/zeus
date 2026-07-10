# Training Zeus

The model can be trained via the `zeus train` CLI command. It can either:

- load an existing pre-trained snapshot and continue training,
- or create a completely new model and train from scratch.


## Training from scratch

To replicate the Camera GrandStaff model, use the following command:

```bash
zeus train \
    --experiment zeus-camera-grandstaff-replication \
    --new_model grand26 \
    --train datasets/grandstaff/samples_distorted.train.pickle \
    --augment "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2" \
    --dev datasets/grandstaff/samples_distorted.dev.pickle \
    --test datasets/grandstaff/samples_distorted.test.pickle \
    --epochs 500 \
    --evaluation_from 10 \
    --evaluation_each 10 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --lr_decay cos \
    --quiet_tf
```

> **Note:** Start with a lower batch size, say 8 and go up and see if you even have enough RAM or GPU memory to run at 64.

To see the training progress, run `./tensorboard` and open `localhost:16006` in the browser.


## Fine-tuning an existing model

Fine-tuning on a new dataset, risks forgetting previously learned data:

> **Note:** Notice the learning rate is 10x smaller than in the example above and also the learning rate decay is disabled. Also note that the finetuning dataset is very small, so 20 epochs is really short. Adjust this according to your dataset size.

```bash
zeus train \
    --experiment zeus-olimpic-ft \
    --model models/zeus-olimpic.model \
    --train datasets/authentic-piano/samples.finetune.pickle \
    --dev datasets/authentic-piano/samples.dev.pickle \
    --test datasets/authentic-piano/samples.test.pickle \
    --epochs 20 \
    --evaluation_from 1 \
    --evaluation_each 1 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --lr_decay none \
    --quiet_tf
```


## Command arguments

The `--experiment` argument provides name of the current training run (experiment), for use in tensorboard and logs.

Then you specify the model to be used. Either load an existing model snapshot via the `--model` path, or specify the architecture to use for a new model via the `--new_model` argument. See the previous two sections for example values.

Then you specify the training dataset(s) via one or many `--train` arguments. If many, then they are are concatenated and shuffled into one big training dataset. This may simulate fine-tuning with replay when the old training dataset is also included in the fine-tuning training process. Provide multiple values to a single `--train` option, not multiple `--train` options (that would cause only the last one to be loaded).

The training data may be augmented using the `--augment` argument, which then defines a set of filters and their parameters. For each sample, each filter is randomly applied with 50:50 change and a value (e.g. rotation amount) is uniformly sampled from a range specified as an argument of the filter. See `construct_tf_dataset.py` for more details.

For evaluation (continuous and terminal) specify the validation and evaluation datasets via `--dev` (validation) and `--test` (evaluation). Again, arguments may be provided multiple times to have multiple validation/evaluation numbers. Validation is run after each epoch, evaluation may only be run every N epochs and then it also runs after training finishes. This is controlled by `--evaluation_from` and `--evaluation_each` (delay start, then evaluate every N epochs).

Training always happens for a fixed number of epochs, `--epochs` specifies their number. If you want to simulate early stopping, simply observer validation loss curves and then pick the appropriate model snapshot. Model weights are saved after each evaluated epoch to allow for this time-travelling. If unsure, set the epoch count higher then you expect, you can always kill the process and use whichever snapshot is the latest.

Batch size is specified via `--batch_size`, use the highest that fits into GPU memory.

Learning rate is specified via `--learning_rate`, which may have a cosine decay if `--lr_decay cos` is used, otherwise the learning rate is constant.

The `--quiet_tf` argument disabled Tensorflow warnings and verbose logs. You can set this flag when training regularly, however, if just starting the model for the first time on an unknown GPU, do NOT use this flag to see from the logs, whether Tensorflow actually uses the GPU provided or not.


## Tips

- Prefix the command with `time` to have the total runtime printed in the standard output.
- Start with a lower batch size, say 8 and go up and see if you even have enough RAM or GPU memory to run at 64.
- Remove `--quiet_tf` when testing a new GPU to see if it's acutally being used.
- Training time is specified with epochs, however, the total training steps then depend on dataset size (and batch size if that's not fixed). When combining multiple datasets and comparing results, adjust epochs to match the training time.
- When fine-tuning, lower learning rate and disable its decay.


## Output files

`logs/{experiment-name}` TODO ...
