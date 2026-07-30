# Evaluation metrics

Zeus measures a model by comparing predicted LMX token sequences against gold ones. Everything here is a symbol error rate over some subset of the tokens; the metrics differ only in which tokens they count.

**SER is the metric.** It is what training reports, what `zeus evaluate` writes, and what the prediction visualization sorts by. The others exist because one number over every token answers only one question, and they are opt-in — asked for per run rather than computed always.


## What there is

| Name | Counts | Use it when |
| --- | --- | --- |
| `SER` | Every token. | Always. This is the development metric and the one to compare between runs. |
| `SERnotuplets` | Every token except tuplet brackets and ratios (`tuplet:start`, `tuplet:stop`, `3in2`, …). | Your corpus has many implicit tuplets that the annotation had to guess at, and they are drowning the real signal. This is why it exists: the OpenScore Lieder Corpus behind the 2024 experiments is full of them. |
| `SERpitchonly` | Pitch tokens alone (`C4`, `A2`, …), ignoring rhythm, clefs, beams and everything else. | You want to know whether the model is reading the right notes off the staff lines, separately from whether it got their durations right. |

All three are percentages, and all three can exceed 100: an insertion is an error but adds nothing to the denominator, so a model that predicts far too much scores above 100 rather than saturating.


## Selecting them

Both `zeus train` and `zeus evaluate` take `--metrics`, a comma-separated list:

```bash
zeus evaluate \
    --model-snapshot models/solo26.model \
    --dataset datasets/omniomr/samples.test.pickle \
    --metrics SER,SERpitchonly
```

```
SER: 4.123
SERpitchonly: 1.902
```

During training the same argument decides what each periodic evaluation reports, both into the log directory and into tensorboard, where each becomes its own curve named after the dataset and the metric.

Names are checked before the model is loaded, so a typo fails immediately rather than after the wait:

```
Unknown metric 'TEDn'. Available metrics are: SER, SERnotuplets, SERpitchonly.
```

Order is kept and repeats are dropped. `zeus visualize-predictions` always uses SER, since the ordering it produces is the point and one ordering is enough.


## From python

```py
from zeus.evaluation import ALL_METRICS, SER, SER_PITCH_ONLY, compute_metrics

SER.compute(gold_lmx_strings, predicted_lmx_strings)  # -> 4.123
compute_metrics([SER, SER_PITCH_ONLY], gold, predicted)  # -> {"SER": …, "SERpitchonly": …}
```

`Zeus.evaluate` and `Zeus.train` both take a `metrics=` argument of the same list, defaulting to `[SER]`.


## How the rate is computed

Edit distance between predicted and gold token sequences, **summed over the whole corpus and divided once at the end**:

```
SER = 100 × (total edit distance) / (total gold tokens)
```

That is not the same as averaging the per-sample rates, and the difference matters: a one-token sample getting its one token wrong is 100%, and averaging would let it outweigh a fifty-token sample that got one token wrong. Summing first weighs every token equally, which is what the literature reports and what makes two runs comparable.

A filtered metric applies its filter to *both* sides. If it did not, a token the metric ignores in the gold would still count as an insertion when the model predicted it.

If nothing anywhere survives the filter — a corpus with no pitches at all, evaluated with `SERpitchonly` — the metric raises rather than returning a number, because there is no rate to report:

```
There are no gold tokens to compare. Either the gold data is empty,
or this metric keeps no tokens that appear in it.
```


## What these numbers do not tell you

They are computed on the token sequence, not on the music. Two transcriptions a musician would call equivalent can differ in LMX — a different but valid voice assignment, a rest spelled as two tied rests — and SER counts that as error.

So SER is a training signal and a way to compare runs, not a claim about musical correctness. Answering that properly means comparing at the MusicXML level, with a tree edit distance or similar, which requires decoding and is expensive enough to belong to a separate benchmarking rig run after a model is trained. It is deliberately not in this repository: Zeus is the image-to-sequence model, and LMX is the sequence.


## Adding a metric

Write it in `src/zeus/evaluation/metrics.py` and name it in `ALL_METRICS`. Nothing else changes — the CLI resolves whatever names that module publishes, and the help text lists them.

A metric that is a symbol error rate over a subset of tokens is a one-liner:

```py
SER_PITCH_ONLY = Metric(
    name="SERpitchonly",
    description="Symbol error rate over pitch tokens alone, …",
    compute=_symbol_error_rate_over(_is_pitch_token),
)
```

A metric that is something else entirely only has to be a function from gold and predicted LMX strings to one number.
