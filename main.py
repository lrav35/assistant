import shutil
import os
from dotenv import load_dotenv
from typing import Optional, Tuple
from mlx_lm import load
import mlx.core as mx
import mlx.nn as nn
import time
from transformers import PreTrainedTokenizer

USER = 'user'
ASSISTANT = 'assistant'

def load_model(model_name: str):
    model, tokenizer = load(model_name, tokenizer_config={"trust_remote_code": True})
    return model, tokenizer

# TODO: move max_tokens to a config file
def generate(model: nn.Module, tokenizer: PreTrainedTokenizer, prompt: str, max_tokens: int, temp: float = 0.0) -> str:
    """Code largely taken from mlx-lm"""
    # encode
    prompt = mx.array(tokenizer.encode(prompt))

    tic = time.perf_counter()
    tokens = []
    token_string = ""
    cur = 0
    REPLACEMENT_CHAR = "\ufffd"

    print("\n")

    for (token, prob), n in zip(generate_step(prompt, model, temp), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if REPLACEMENT_CHAR not in tokens:
            t = s[cur:]
            token_string += t
            print(colored(t, os.getenv("COLOR")), end="", flush=True)
            cur = len(s)
    print("\n")
    gen_time = time.perf_counter() - tic
    gen_tps = (len(tokens) - 1) / gen_time
    print(f"token count: {len(tokens) + len(prompt)}")
    print(f"tokens-per-sec: {gen_tps:.3f}\n")

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


def colored(st, color: Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st

def spacers(): return "\n" + "#" * shutil.get_terminal_size().columns + "\n" 

def print_file_content(filename: str):
    print("\n")
    with open(filename, 'r') as file:
        for line in file:
            print(line, end='')
            time.sleep(0.02)

def format_tokens(tokens: str, being: str):
    prefix = os.getenv('USER_PREFIX') if being == USER else os.getenv('ASSISTANT_PREFIX')
    token_string = prefix + "\n" + tokens + os.getenv('EOS')
    return token_string

def run():
    load_dotenv()
    print_file_content("shoggoth.txt")

    print("\n\nBeginning to load model...\n" + spacers())

    model, tokenizer = load_model(os.getenv("MODEL"))

    print(spacers())

    print("what would you like to know? ('quit' + <Enter> to exit)\n")

    outputted = ""

    while 1:
        toks = input(">>> ")
        if toks == "quit": 
            print(colored("\nprogram ending... (╥ꞈ╥)", "red"))
            break
        while 1:
            cur = format_tokens(toks, USER)
            outputted += cur
            outputted = format_tokens(generate(model, tokenizer, outputted, 1000), ASSISTANT)
            break
    print("\n")

if __name__ == "__main__":
    run()