"""Running Zeus as a Musibot *Model*.

[Musibot](https://github.com/OmniOMR/musibot) deploys OMR models as isolated
subprocesses. A *Worker Head* starts one, hands it two file descriptors, and
drives it with JSON lines; the images and transcriptions travel through a
directory. That is the whole of what Musibot asks, and it is what
`zeus musibot` implements.

The process boundary is the point. Zeus needs python 3.10 and TensorFlow 2.12,
while a *Worker Head* needs python 3.11 or newer, so the two could not share an
environment even in principle — and across pipes they do not have to.

Nothing in this package imports anything from Musibot: a *Model* speaks the
protocol and touches files, and carries no Musibot dependency at all.
"""
