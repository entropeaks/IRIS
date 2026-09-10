"""Geometric augmentation that keeps a segmented dataset segmented.

`augment_dataset.py` multiplies a dataset by pasting each cap onto random web
backgrounds, which is one answer to background nuisance. Segmenting to a flat
grey is the competing answer, and the two cannot be composed: replacing the
background of a segmented image undoes the segmentation.

So a segmented dataset needs its own augmentation, and the constraint is that
every transform must leave the background flat and the same colour. Rotations
and rescales fill their empty corners with the background colour sampled from
the source image, so the output is still segmented and a descriptor fitted on
it still describes the same distribution as the gallery it will be scored
against -- which is the whole point. Measured on `augmented_data16`, a train
split from the wrong distribution takes a frozen ZCA pipeline from 0.647 R@1
to 0.069; distribution match matters more than sample count.

Variant 0 of each image is the image itself, so the originals are always in
the output and the multiplier counts total images per original, not extras.

    python scripts/augment_segmented_dataset.py \
        --source /path/to/seg_gray --target /path/to/seg_gray_aug16 --multiplier 4
"""

import argparse
import random
from pathlib import Path

from PIL import Image


def background_colour(img: Image.Image) -> tuple:
    """The flat background, read off the corners rather than assumed.

    Sampling keeps this usable on seg_black as much as seg_gray, and a corner
    that disagrees with the others means the image is not segmented -- better
    to fill with something close than to hardcode a grey that is wrong.
    """
    w, h = img.size
    corners = [img.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    channels = zip(*corners)
    return tuple(sorted(values)[len(values) // 2] for values in map(list, channels))


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """One geometric variant, background preserved in shape and in value.

    Geometric only. A brightness or contrast jitter would move the flat
    background off the value the gallery has -- measured, it drifted from 128
    to between 111 and 141 -- which puts a variance direction in the training
    corpus that the gallery does not have, and hands anything fitted there a
    nuisance to suppress that never existed. The point of this script is to add
    quantity without moving the distribution, so it adds nothing else.
    """
    fill = background_colour(img)

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # a free angle rather than a multiple of 90: the lossless quarter turns are
    # already handled at inference by the rotation averaging, so repeating them
    # here would add copies the descriptor is invariant to anyway
    img = img.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, fillcolor=fill)

    # scale jitter, cropping in or padding out around the centre
    side = img.size[0]
    scale = rng.uniform(0.85, 1.15)
    scaled = img.resize((max(1, round(side * scale)),) * 2, Image.BICUBIC)
    canvas = Image.new(img.mode, (side, side), fill)
    offset = (side - scaled.size[0]) // 2
    canvas.paste(scaled, (offset, offset))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--multiplier", type=int, default=4,
                        help="images written per source image, the original included")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    written = 0
    for class_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        out_dir = args.target / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in class_dir.iterdir()
                                 if not p.name.startswith(".")):
            img = Image.open(image_path).convert("RGB")
            for n in range(args.multiplier):
                variant = img if n == 0 else augment(img, rng)
                variant.save(out_dir / f"{n}-{image_path.name}")
                written += 1

    print(f"wrote {written} images to {args.target}")


if __name__ == "__main__":
    main()
