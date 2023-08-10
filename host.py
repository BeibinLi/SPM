from flask import Flask, request
import argparse
from model_utils import load_inference_model, answer
from termcolor import colored

app = Flask(__name__)


def get_args() -> argparse.Namespace:
    """Parses command line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        default=None,
                        help="Dir to load model")
    parser.add_argument("--cuda_id",
                        type=int,
                        default=0,
                        help="Which cuda to load model.")
    return parser.parse_args()


class Host:

    def __init__(self, args):
        self.args = args
        self.tokenizer, self.config, self.tuned_model = load_inference_model(
            args.dir, False)
        _, _, self.original_model = load_inference_model(args.dir, True)

    def chat(self):
        message = request.json.get('message')
        model_name = request.json.get("model", "origin")
        max_tokens = request.json.get("max_tokens", 1000)
        temperature = request.json.get("temperature", 0)
        messages = request.json.get("messages", [])
        top_p = request.json.get("top_p", 0.7)
        n = request.json.get("n", 1)
        secret = request.json.get("secret", None)

        if model_name not in ["tuned", "origin"]:
            return {
                'answers': [
                    "Model name is wrong! It should be "
                    "either 'tuned' or 'origin'"
                ]
            }

        if secret not in ["secret", "API KEY"]:
            return {'answers': ["API secret is wrong!"]}

        try:
            _answers = answer(
                message,
                tokenizer=self.tokenizer,
                model=self.tuned_model
                if model_name == "tuned" else self.original_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                top_p=top_p,
                num_return_sequences=n,
            )
        except Exception as e:
            _answers = [f"Runtime Error {type(e)}: {e}"]

        print("Return:", colored(_answers, "green"))
        return {'answers': _answers}


if __name__ == '__main__':
    host = Host(get_args())
    app.add_url_rule('/chat', view_func=host.chat, methods=['POST'])
    app.run(host='localhost', port=5000)
