# lv-serve

Llama 3.2 Vision OpenAI-like API server. Supports `/v1/chat/completions` and `/v1/models` endpoints, image and text messages.

This is just a toy, a POC, an exercise in learning how to connect 🤗Transformers with FastAPI.

The API server is very much based off of the one from <https://github.com/sam-paech/antislop-sampler>. Go check out their project! I just stripped it down to the aforementioned endpoints, replaced the model/tokenizer with Llama 3.2 Vision, and wrote the image extraction code. It certainly saved a lot of time as I'd been trying to tackle this project by starting with [OpenAI's OpenAPI spec for their API](https://github.com/openai/openai-openapi), trimming *that* down, and then running it through a Swagger/OpenAPI code generator.

And yes, I'm aware of [vllm](https://github.com/vllm-project/vllm) but my 16GB GPU is on a Windows rig, and I'm not too keen on doing LLM stuff in WSL2 Docker.

My original goal for this project was to run it on one of my home servers that happened to have an 8GB GPU. But neither this project nor vllm could work with so little VRAM. Why is CPU offloading so hard? GGUF when???

Anyway...

## Features

* [x] `/v1/chat/completions` and `/v1/models` endpoints
* [x] The chat endpoint supports streaming and non-streaming
* [x] Messages may contain images, either as a remote URL or [base-64 encoded](https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data)
* [x] Sampler settings
* [ ] Setting the seed

## Installation

If you're using Python 3.12 + CUDA 12.4, you can probably use my requirements file directly:

    pip install -r requirements.txt

Otherwise, try the unpinned one:

    pip install -r requirements.in

Or if you're familiar with [pip-tools](https://github.com/jazzband/pip-tools), you can just regenerate `requirements.txt` for whatever Python version/platform you're using.

## Running

    python run_api.py --model <model-ID>

`model-ID` can be a path to a local model, or the repo-ID of a Llama 3.2 Vision model on Huggingface. For 4-bit quantized, there is currently:

* `SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4` ([link](https://huggingface.co/SeanScripts/Llama-3.2-11B-Vision-Instruct-nf4))
* `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` ([link](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit))

They will require approximately 10GB VRAM to run. (I included a sample script to quantize the model, if that's something you'd rather do yourself.)

By default, the server will listen on all interfaces on port 8000, but this can be changed with the `--host` and `--port` options.

You can also choose to apply `bitsandbytes` 4-bit/8-bit quantizing as you load the original model. Simply add the `--load_in_4bit` or `--load_in_8bit` options:

    python run_api.py --model meta-llama/Llama-3.2-11B-Vision-Instruct --load_in_4bit

(Note that `--load_in_8bit` will require around 15GB of VRAM, and for some reason, is significantly slower than 4-bit.)

Do not use `--load_in_4bit` or `--load_in_8bit` when already using a pre-quantized model, though at worst, you'll just get an extra warning from the `transformers` library.

## License

Licensed under [Apache-2.0](https://opensource.org/license/apache-2-0).
