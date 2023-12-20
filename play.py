#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import PIL.Image as Image
import PIL.ImageOps as IOps
from numpy.typing import NDArray

from cmo_digits.network import Network

DEFAULT_MODEL = Path("./_models/Sigmoid_784,30,10_3.0_10_30.pkl")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("image", type=Path, metavar="IMAGE", help="image to predict")
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        help="path to the model to use",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--acts", "-a", action="store_true", help="print activations")

    return parser


def to_input(filepath: Path) -> NDArray[np.float64]:
    img = Image.open(filepath)
    img = IOps.grayscale(img).resize((28, 28))

    return np.array(img).reshape((784, 1)) / 255


def main():
    args = get_parser().parse_args()

    if not args.model.exists():
        print("model doesn't exist")
        sys.exit(1)

    net = Network.from_pkl(args.model)
    acts = net.feedforward(to_input(args.image))

    if args.acts:
        print(f"activations:\n{acts}")
    print(f"is it: {acts.argmax()}?")


if __name__ == "__main__":
    main()
