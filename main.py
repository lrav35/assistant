import sys
from typing import Optional




def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st

def output(outputted, tokens, color):
    # this is where we will decode tokens from the model and build the context for the next prompt
    cur = tokens
    context = outputted + cur
    sys.stdout.write("current input: " + colored(cur, color)+"\n")
    sys.stdout.write("all context: " + colored(context, color)+"\n")
    sys.stdout.flush()
    outputted += cur
    return outputted

def run():
    print("insert loading model here...")

    print("what would you like to know?")

    toks = ""
    outputted = output("", toks, "green")

    while 1:
        toks = input(">>> ")
        while 1:
            outputted = output(outputted, toks, "blue")
            break
    print("")

if __name__ == "__main__":
    run()