import glob
import os
import json
from termcolor import colored
import argparse
import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
    """Parses command line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Dir to load model")
    return parser.parse_args()


def load_latest_state(experiment_dir):
    """
    Loads the latest trainer state from the given experiment directory.

    Args:
        experiment_dir (str): Path to the experiment directory containing the
             model's checkpoint.

    Returns:
        trainer_state (json): Trainer state loaded from the checkpoint.
    """
    latest_checkpoint = max(glob.glob(
        os.path.join(experiment_dir, "checkpoint-*")),
                            key=os.path.getctime)
    print(colored(f"Loading model from {latest_checkpoint}", "yellow"))
    return json.load(open(latest_checkpoint + "/trainer_state.json", "r"))


if __name__ == "__main__":
    args = get_args()

    trainer_state = load_latest_state(args.dir)
    log_history = trainer_state["log_history"]

    x, y = [], []
    for log in log_history:
        x.append(log["step"])
        y.append(log["loss"])

    plt.plot(x, y)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
