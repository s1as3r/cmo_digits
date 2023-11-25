import json
import time
from argparse import ONE_OR_MORE, ZERO_OR_MORE, Action, ArgumentParser
from pathlib import Path

from cmo_digits.activation.leaky_relu import LReLU
from cmo_digits.activation.relu import ReLU
from cmo_digits.activation.sigmoid import Sigmoid
from cmo_digits.activation.tanh import Tanh
from cmo_digits.network import Network
from cmo_digits.utils import load_data

DIGIT_HEIGHT = 28
DIGIT_WIDTH = 28

OUTPUT_NEURONS = 10

ACTIVATION_FNS = {"sigmoid": Sigmoid, "relu": ReLU, "lrelu": LReLU, "tanh": Tanh}

DATASET = "./datasets/mnist.npz"


def main():
    parser = get_parser()
    args = parser.parse_args()

    sizes = [DIGIT_HEIGHT * DIGIT_WIDTH, *args.layers, OUTPUT_NEURONS]
    activation_fn = ACTIVATION_FNS.get(args.activation.lower(), Sigmoid)(
        **(args.act_args or {})
    )

    net = Network(sizes, activation_fn)

    training_data, testing_data = load_data(DATASET)

    print("Training Network With:")
    print(f"\tlayers = {sizes}")
    print(f"\tn_epochs = {args.epochs}")
    print(f"\tmini-batch size = {args.batch_size}")
    print(f"\tlearning rate = {args.eta}")
    print(f"\tactivation function = {type(activation_fn).__name__}")
    if args.act_args:
        print(f"\tactivation fn args = {args.act_args}")

    start_time = time.time()
    net.stochastic_gd(
        training_data,
        args.epochs,
        args.batch_size,
        args.eta,
        testing_data,
    )
    took = int(time.time() - start_time)
    print(f"took: {took} seconds")

    if args.save_model:
        if args.save_model.exists():
            print(f"{args.save_model} already exists")
        else:
            net.save_to_pkl(args.save_model)

    if args.save_stats:
        if args.save_stats.exists():
            print(f"{args.save_stats} already exists")
        else:
            stats = {
                "epochs": args.epochs,
                "mini_batch_size": args.batch_size,
                "eta": args.eta,
                "layers": sizes,
                "activation_fn": activation_fn.__name__,
                "time_taken": took,
                "accuracy": net.accuracy,
                "activation_fn_args": args.act_args,
            }

            with open(args.save_stats, "w") as f:
                json.dump(stats, f)


class ParseKwargs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split("=")

            try:
                value = float(value)
            except ValueError:
                pass

            getattr(namespace, self.dest)[key] = value


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="cmo_digits training",
        description="utitlity script for training the model with different parameters",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        metavar="N_EPOCHS",
        default=30,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        metavar="BATCH_SIZE",
        default=10,
        help="mini-batch size",
    )
    parser.add_argument(
        "--eta", "-r", type=float, metavar="RATE", default=3.0, help="learning rate"
    )
    parser.add_argument(
        "--layers",
        "-l",
        nargs=ONE_OR_MORE,
        type=int,
        metavar="NEURONS",
        default=[30],
        help="neurons in respective hidden layers",
    )
    parser.add_argument(
        "--activation",
        "-a",
        choices=ACTIVATION_FNS.keys(),
        default="sigmoid",
        help="activation function to use",
    )
    parser.add_argument(
        "--save-model",
        "-s",
        type=Path,
        metavar="FILEPATH",
        help="save the model to FILEPATH",
    )

    parser.add_argument(
        "--save-stats",
        type=Path,
        metavar="FILEPATH",
        help="save the model statistics to a json file",
    )

    parser.add_argument(
        "--act-args",
        action=ParseKwargs,
        nargs=ZERO_OR_MORE,
        metavar="PARAM=VALUE",
        help="pass arguments to the activation function",
    )

    return parser


if __name__ == "__main__":
    main()
