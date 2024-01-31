import sys
from typing import Optional, Tuple
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn
import time
from transformers import PreTrainedTokenizer


def load_model():
    model, tokenizer = load("stabilityai/stablelm-2-zephyr-1_6b")
    return model, tokenizer

# TODO: move max_tokens to a config file
def generate(model: nn.Module, tokenizer: PreTrainedTokenizer, prompt: str, max_tokens: int, temp: float = 0.0) -> str:
    """Code largely taken from mlx-lm"""
    # encode
    prompt = mx.array(tokenizer.encode(prompt))

    # TODO: add back toks/s if i want
    # tic = time.perf_counter()
    tokens = []
    REPLACEMENT_CHAR = "\ufffd"

    # send to model
    for (token, prob), n in zip(generate_step(prompt, model, temp), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        # if n == 0:
        #     prompt_time = time.perf_counter() - tic
        #     tic = time.perf_counter()
        tokens.append(token.item())

    token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")

    return token_string

def generate_step(prompt: mx.array, model: nn.Module, temp: float = 0.0):
    """Code largely taken from mlx-lm"""
 
    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y, prob = sample(logits)
        yield y, prob


def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st

def output(outputted, tokens, color):
    # this is where we will decode tokens from the model and build the context for the next prompt

# <|user|>
# Which famous math number begins with 1.6 ...?<|endoftext|>
# <|assistant|>
# The number you are referring to is 1.618033988749895. This is the famous value known as the golden ratio<|endoftext|>

    cur = tokens
    context = outputted + cur
    sys.stdout.write("current input: " + colored(cur, color)+"\n")
    sys.stdout.write("all context: " + colored(context, color)+"\n")
    sys.stdout.flush()
    outputted += cur
    return outputted

def run():
    model, tokenizer = load_model()

    print("what would you like to know?")

    toks = ""
    outputted = output("", toks, "green")

    while 1:
        toks = input(">>> ")
        while 1:
            outputted = output(outputted, toks, "blue")
            response = generate(model, tokenizer, toks, 100)
            print(response)
            break
    print("")

if __name__ == "__main__":
    run()