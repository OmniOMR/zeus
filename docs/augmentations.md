# Augmentations

This page documents the `--augment` CLI option value. In other words, which training data augmentations there are and what arguments they have.

Let's take a sample augmentation pipeline (and a reasonable one you might use as-is):

```bash
--augment "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2"
```

It is a comma-separated list of filters, which together form a sequential pipeline. Augmentation are run as the very last thing before the sample is fed to the model, which means the input image is already normalized height-wise to the model, is grayscale and possible width-squished if too wide.

For each sample the augmentation pipeline chooses with a 50:50 chance whether a given augmentation filter will be applied or skipped. Then those filters that remain are sequentially applied to the image.

Each filter is a filter name followed by colon-separated arguments (`name:arg1:arg2`).

This is the list of available filters and their parameters:

- `h:8` Horizontal shift by at most `8` pixels left or right
- `rotate:1` Rotation by at most `1` degree in either direction
- `v:4` Vertical shift by at most `4` pixels up or down
- `de` Dilatation/erosion in a random direction on an ellipse with x semi-axis 1 and y semi-axis 0.5
- `en3:0.2` For a random probability of up to `0.2`, negate pixels whose value and value of their 8 neighbors are not uniformly white or uniformly black (boundary-sensitive noise)
- `n:0.01` Negate a pixel with a probability of `0.01`, independently for each pixel in the image (global noise)
- `c:-1:1` Adjust contrast by a random factor in `-1` to `1` range
- `b:-0.5:0.2` Adjust brightness by a random factor in `-0.5` to `0.2` range
