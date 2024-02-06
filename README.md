<!-- ABOUT THE PROJECT -->
## Assistant


I want the ability to interface with an omniscient being directly on my laptop. I do not want to be reliant on some company's black box of an API. If I have the weights, I don't want to pay to use A100s and short of a GPU cluster in my basement, this is the next best thing. The introduction of the [MLX](https://github.com/ml-explore/mlx) framework has given engineers the ability to run models locally on their Apple silicon-based machines, similarly to the [llama.cpp](https://github.com/ggerganov/llama.cpp) ecosystem with GGUF. This is a *very* simple python application that will allow me to do this. For now, this is only viable with quantized or very small models depending on the specs of your machine

<video src="https://github.com/lrav35/assistant/assets/49992169/0e33505e-0a15-404f-b794-2e10f084ccda" controls="controls" style="max-width: 730px;">
</video>

<!-- GETTING STARTED -->
## Getting Started


This project was built with python 3.9 and has been tested on my Macbook Pro M2 w/ 16GB of unified RAM.

### Usage

1. Clone the repository 
2. Create a virtual env
    ```
    python -m venv .venv
    ```
3. Install the few dependencies
    ```
    pip install -r requirements.txt
    ```
4. Run application
    ```
    python main.py
    ```

Configuration can be found int the [.env](.env) file. These are environment variables that you can use to changes the model and other small aspects about the application

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
