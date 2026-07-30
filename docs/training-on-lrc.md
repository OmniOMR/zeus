# Training on the UFAL LRC cluster

At Charles University, Institute of Formal and Applied Linguistics we have a compute cluster called LRC. This documentation page describes the setup that can be used to train Zeus there.

The documentation for the cluster is behind the linguistic password at: https://ufal.mff.cuni.cz/lrc

The older wiki documentation page is here: https://wiki.ufal.ms.mff.cuni.cz/slurm


## Getting a python 3.10 venv

To set up a venv for Zeus, use this Python executable:

```
/opt/python/3.10.7/bin/python3
```

> **Note:** There have been some issues with it recently (July 2026) as machines are being updated to newer Ubuntu. Access it from `lrc1` or other cluster nodes. Ordinary machines in the network may fail with `version GLIBC_2.38 not found` (e.g. `geri` does that).


## Suitable GPU machines

The list of GPU machines is here: https://ufal.mff.cuni.cz/lrc/index.php?title=Environment

There are a number of GPU machines in the cluster, but Zeus is quite an old model and does not need the latest and hottest hardware to train. I have tested training it on `A40`, `L40`, `A30`, and `RTX 3090`. I can select these cards by providing `--constraint="gpuram48G|gpuram24G"` to an `srun` command.

Zeus likely works on other GPUs as well, but these constraints give me enough to train already.

In terms of RAM, I tested that 32GB is enough to keep all the necessary data in memory and train.

Before running the train command though, make sure your session has CUDA enabled by running:

```bash
module load cuda/11.8-cudnn8.6
```

This just adds the proper CUDA version to `PATH` and `LD_LIBRARY_PATH`.


## Getting an interactive session

This is the command to get a usable interactive job:

```bash
srun --pty --gpus=1 --mem=32G -p gpu-ms,gpu-troja --constraint="gpuram48G|gpuram24G" bash

module load cuda/11.8-cudnn8.6

.venv/bin/zeus train \
    ...
```


## Sbatch script housekeeping

After testing the train command does not crash from an interactive job, you can run the whole training via `sbatch`. For that you need the command to be placed in a `.sh` file somewhere.

I create a `/slurm` folder in the root of my working directory and then run my scripts like this:

```bash
sbatch slurm/my-experiment.sh
```

Here are example contents of `slurm/zod-bw-auth.sh`:

> **Note:** `zod-bw-auth` stands for "Zeus on Dolores: B/W and Authentic data".

```bash
#!/bin/bash
#SBATCH -J zod-bw-auth
#SBATCH -p gpu-ms,gpu-troja
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --constraint="gpuram48G|gpuram24G"
#SBATCH --mem=32G
#SBATCH -o logs/slurm-%x.%j.out
#SBATCH -e logs/slurm-%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mayer@ufal.mff.cuni.cz
#SBATCH --time="14-0"

# make cuda available
module load cuda/11.8-cudnn8.6

time .venv-zeus/bin/zeus train \
    --experiment zod-bw-auth \
    --new_model solo26 \
    --train \
        datasets/dolores/samples_rendered.train.pickle \
        datasets/dolores/samples_rendered.test.pickle \
        datasets/dolores/samples.train.pickle \
        datasets/dolores/samples.test.pickle \
    --augment "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2" \
    --dev \
        datasets/dolores/samples_rendered.validation.pickle \
        datasets/dolores/samples.validation.pickle \
    --test \
        datasets/omniomr/samples_rendered.test.pickle \
        datasets/omniomr/samples.test.pickle \
    --epochs 430 \
    --evaluation_from 10 \
    --evaluation_each 10 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --lr_decay cos \
    --quiet_tf
```

Every line in this file is hard-earned. Let's explore:

- `SBATCH -J` is the job name, keep it the same as the `--experiment` name for Zeus and also the same as the name of this `.sh` file. So experiment `foo` will be started as `sbatch slurm/foo.sh`, will run in slurm as `foo` and will log to `logs/foo-{timestamp}`
- `SBATCH -p` makes sure we use GPUs from both MS and Troja
- `SBATCH -N 1` just one job instance
- `SBATCH --gpus=1` the job needs one GPU
- `SBATCH --constraint="gpuram48G|gpuram24G"` we want GPUs we've tested against
- `SBATCH --mem=32G` enough RAM to load datasets
- `SBATCH -o logs/slurm-%x.%j.out` write output to the logs folder, next to the log folder produced by the zeus train command. `%x` is job name and `%j` is job number assigned by slurm on startup.
- `SBATCH -e logs/slurm-%x.%j.err` same thing for errors
- `SBATCH --mail-type=ALL` send emails about startup/crash/completed
- `SBATCH --mail-user=mayer@ufal.mff.cuni.cz` to this email address
- `#SBATCH --time="14-0"` increase job timeout to 14 days. Defaults to 7 days. The script above should, however, complete in 3-4 days.

**What to replace when you copy this file:**

- Rename the file to your experiment name and set `SRUN -J` and `--experiment` to the same new name.
- Replace my email address with yours or drop the email block entirely.


## Tensorboard

The offline version:

1. Copy the logs folder from cluster to your laptop
2. Run tensorboard locally `.venv/bin/tensorboar --logdir logs`
3. Open browser at `http://localhost:6006`

The real-time version:

The trick is to run tensorboard on `sol1` (because `lrc` lacks AVX 512 and tensorboard crashes there) and SSH-tunnel it through `blackbird` to your laptop.

```bash
ssh -L 6006:localhost:6006 mayer@blackbird.ms.mff.cuni.cz
ssh -L 6006:localhost:6006 sol1
cd your-folder-with-the-work
.venv/bin/tensorboard --logdir logs --port 6006
```

The port `6006` is the default for tensorboard, however, if there's anyone already running such a tunnel through the same machines, you will not be allowed to take it. So either use different machines (`sol[1-8]` and `geri|freki|blackbird`) or use a different port (say `6007` or `16006`).
