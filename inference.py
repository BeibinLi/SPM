import argparse
import os
import pdb

from model_utils import answer, load_inference_model, load_latest_model
from termcolor import colored
from utils import get_spm_dataset


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
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help=
        "Mode: 'auto' for auto testing on random samples, 'manual' for manual input."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="finetune",
        choices=["finetune", "pretrain"],
        help="Dataset to use: 'finetune' for finetune dataset, 'pretrain' for "
        "pretrain dataset. If set to 'finetune', the output with wrong "
        "classification will be rectified.")
    args = parser.parse_args()
    args.dir = os.path.abspath(os.path.expanduser(args.dir))
    print(args)
    return args


if __name__ == "__main__":
    args = get_args()

    tokenizer, config, model = load_inference_model(args.dir)

    # question = "hi"
    # pdb.set_trace()

    if args.mode == "manual":
        while True:
            print("-" * 30)
            question = input("Human: ").strip()

            if not question:
                continue
            if question == "quit":
                break
            elif question == "pdb":
                pdb.set_trace()
            elif question == "load":
                # Reload the model and config
                config, model = load_latest_model(model, args.dir)
            else:
                ans = answer(question, tokenizer, model)[0]
                print("Bot:", colored(ans, "green"))
    else:
        test_dataset = get_spm_dataset(phase="finetune",
                                       mode="test",
                                       with_self_instruct=True)
        for i in range(20):
            text = test_dataset[i]["text"].split("### Human:")[-1].strip()
            input, std = text.split(
                "### Assistant:") if "### Assistant:" in text else (text, "")
            input, std = input.strip(), std.strip()
            output = answer(input, tokenizer, model)[0]

            print("-" * 30)
            print("Input:", input)
            print("Output:", colored(output, "green"))

            if args.dataset == "finetune" and output[:5] != std[:5]:
                output = answer(input,
                                tokenizer,
                                model,
                                rectifier=std[:5],
                                verbose=False)[0]
                print("Rectified output:", std[:5], colored(output, "yellow"))
            print("GT output:", colored(std, "blue"))
