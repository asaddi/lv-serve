# Originally from https://github.com/sam-paech/antislop-sampler @ e931054 || Apache-2.0
# Modified for Llama 3.2 Vision by Allan Saddi <allan@saddi.com>
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
import queue
import time
from typing import Generator, Iterator, List, Dict, Optional, Any
import urllib
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging
import PIL.Image
from pydantic import BaseModel, Field
import threading
import torch
from transformers import (
    AutoProcessor,
    BatchEncoding,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

# Global variables to hold the model and tokenizer
model: Optional[PreTrainedModel] = None
processor: Optional[PreTrainedTokenizer] = None

# Variables to store model metadata
model_loaded_time: Optional[int] = None
model_name_loaded: Optional[str] = None

default_max_tokens = 2048

# Thread pool for running the model's generate function
generate_thread_pool: ThreadPoolExecutor|None = None

# Define a global threading.Lock to enforce single concurrency
lock = threading.Lock()

# Define Pydantic models for request and response schemas

class ChatMessagePartText(BaseModel):
    type: str = 'text'
    text: str


class ChatMessageImageUrl(BaseModel):
    url: str


class ChatMessagePartImageUrl(BaseModel):
    type: str = 'image_url'
    image_url: ChatMessageImageUrl


# This one is only used internally, and might be specific to MllamaProcessor?
class ChatMessagePartImage(BaseModel):
    type: str = 'image'


class ChatCompletionMessage(BaseModel):
    role: str
    content: str|list[ChatMessagePartText|ChatMessagePartImageUrl|ChatMessagePartImage]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model to use for completion")
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=None, ge=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-K sampling")
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum probability threshold")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress")


class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage
    index: int
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# New Pydantic models for /v1/models endpoint

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Any] = []
    root: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Startup event to load model and tokenizer
@asynccontextmanager
async def setup_teardown(_: FastAPI):
    global processor, model
    global model_loaded_time, model_name_loaded, default_max_tokens, generate_thread_pool

    # Load configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    load_in_8bit = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    default_max_tokens = int(os.environ['MAX_TOKENS'])

    # Validate mutually exclusive flags
    if load_in_4bit and load_in_8bit:
        logger.error("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")
        raise ValueError("Cannot set both LOAD_IN_4BIT and LOAD_IN_8BIT. Choose one.")

    # Load tokenizer
    logger.info(f"Loading tokenizer for model '{model_name}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error("Error loading tokenizer:", e)
        raise e

    # Load model with appropriate precision
    logger.info(f"Loading model '{model_name}'...")
    try:
        dtype = torch.bfloat16
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            # Will get warnings about bfloat16 being cast to float16
            dtype = torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # Apparently, it doesn't like this layer being quantized
                llm_int8_skip_modules=['multi_modal_projector'],
            )

        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map='auto',
            quantization_config=quantization_config
        ).eval()
        logger.info("Model loaded.")
    except Exception as e:
        logger.error("Error loading model:", e)
        raise e  # Let FastAPI handle the startup failure

    logger.info("Model and tokenizer loaded successfully.")

    # Store model metadata
    model_loaded_time = current_timestamp()
    model_name_loaded = model_name

    # Set up thread pool used for generation
    generate_thread_pool = ThreadPoolExecutor()
    # Also make it the default, for any future uses of asyncio.to_thread()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(generate_thread_pool)

    try:
        yield
    finally:
        # Cleanup
        if generate_thread_pool is not None:
            generate_thread_pool.shutdown()


app = FastAPI(title="Llama Vision OpenAI-Compatible API", lifespan=setup_teardown)


# Utility functions

def generate_id() -> str:
    return str(uuid.uuid4())


def current_timestamp() -> int:
    return int(time.time())


# Utility function for streaming responses
def stream_tokens_sse(generator: Iterator[str]) -> Generator[str, None, None]:
    """
    Converts text stream to SSE data events.
    """
    text_iter = iter(generator)
    while True:
        try:
            text = next(text_iter)
        except StopIteration:
            # Send final finish_reason to indicate the end of the stream
            finish_data = {
                "choices": [
                    {
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(finish_data)}\n\n"
            logger.debug("Finished streaming tokens.")
            break
        except queue.Empty:
            # This means we timed out
            # Just yield an SSE "comment" to keep the connection alive
            yield ": I'm still alive...\n\n"
        else:
            # Prepare the data in OpenAI's streaming format
            data = {
                "choices": [
                    {
                        "delta": {"content": text},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

            # Yield the formatted data as a Server-Sent Event (SSE)
            yield f"data: {json.dumps(data)}\n\n"
            logger.debug("Yielded token to client.")


async def resolve_image_url(url: str, images: list[PIL.Image.Image]) -> None:
    """
    Given an URL (which may be a data URL), decode/fetch it and add to the
    list of images.
    """
    if url.startswith('data:'):
        # We'll rely on urllib to properly parse it
        with urllib.request.urlopen(url) as resp:
            img = PIL.Image.open(resp) # TODO list acceptable types?
            img.load()
            # Maybe capture the MIME type too?
        images.append(img)
    else:
        # TODO Limit to http/https?
        def fetch_image(url):
            with urllib.request.urlopen(url) as resp:
                img = PIL.Image.open(resp) # TODO list acceptable types?
                img.load()
                # Maybe capture the MIME type too?
            return img
        img = await asyncio.to_thread(fetch_image, url)
        images.append(img)


async def convert_multi_part(parts: list[ChatMessagePartText|ChatMessagePartImageUrl|ChatMessagePartImage], images: list[PIL.Image.Image]) -> None:
    """
    Scans each part of a multi-part message and converts parts of
    type=image_url to type=image as required by MllamaProcessor.

    Decoded/fetched images are appended to the `images` list.
    """
    for index in range(len(parts)):
        part = parts[index]
        if part.type == 'image_url' and isinstance(part, ChatMessagePartImageUrl):
            url_part: ChatMessagePartImageUrl = part
            url = url_part.image_url.url
            if not url:
                raise HTTPException(status_code=400, detail='image_url missing')
            await resolve_image_url(url, images)
            # Convert this part to type=image
            parts[index] = ChatMessagePartImage()


async def extract_images(messages: list[ChatCompletionMessage]) -> list[PIL.Image.Image]:
    """
    Resolve any `image_url` parts in `messages`, converting them to `image`.

    Returns list of extracted images as PIL Images.

    `messages` may be modified in-place, as described above.
    """
    # I believe Llama 3.2 Vision only supports a single image, but we'll
    # build a list anyway.
    images: list[PIL.Image.Image] = []
    for msg in messages:
        # Only valid for user role
        if msg.role == 'user':
            content = msg.content
            if not isinstance(content, str):
                # Not a plain str, assume it's a list or other iterable
                await convert_multi_part(content, images)
    return images


def get_sampler_settings(request: ChatCompletionRequest) -> dict[str,int|float]:
    result = {}
    for k in ('temperature', 'top_k', 'top_p', 'min_p'):
        value = getattr(request, k, None)
        if value is not None:
            result[k] = value
    return result


def generate_common(inputs: BatchEncoding, generate_kwargs: dict[str,Any], streamer: TextIteratorStreamer|None = None):
    assert model is not None

    with torch.no_grad():
        start_ns = time.perf_counter_ns()
        output = model.generate(**generate_kwargs, streamer=streamer)
        end_ns = time.perf_counter_ns()

    # output tensor includes the prompt, so skip over it
    prompt_len = inputs.input_ids.shape[-1]
    generated_ids = output[:, prompt_len:]
    # it's a batch of 1, just grab the main result
    generated_tokens = generated_ids[0]

    gen_time = (end_ns - start_ns) / 1e9
    gen_len = generated_tokens.shape[-1]

    logger.info(f"Generation done. Prompt length = {prompt_len} tokens, generated token count = {gen_len}, total generation time = {gen_time:.2f} s")
    logger.info(f"Average (prompt processing + generation) speed: {(prompt_len+gen_len) / gen_time:.1f} t/s")

    return generated_tokens


# Endpoint: /v1/chat/completions
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, req: Request):
    logger.info("Chat completion request received, waiting for processing...")
    try:
        if model is None or processor is None:
            logger.error("Model and tokenizer are not loaded.")
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")
        assert generate_thread_pool is not None

        # Use the model specified in the request or default
        used_model = request.model if request.model else model_name_loaded

        # Build the prompt from chat messages
        images = await extract_images(request.messages)
        prompt = processor.apply_chat_template(request.messages, add_generation_prompt=True)
        logger.debug(f"Constructed prompt from messages: {prompt}")

        img_list = None if len(images) == 0 else images # MllamaProcessor does not like empty image lists. Use None instead.
        inputs = processor(img_list, prompt, return_tensors='pt').to(model.device)
        prompt_len = inputs.input_ids.shape[-1]

        generate_kwargs = dict(**inputs, max_new_tokens=(default_max_tokens if request.max_tokens is None else request.max_tokens))
        generate_kwargs.update(get_sampler_settings(request))

        if request.stream:
            logger.info("Streaming chat completion request started.")
            # Streaming response

            # Why isn't this documented outside of the source?!
            generator_source = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True, timeout=10)

            def generate_stream():
                try:
                    with lock: # Serialize generation
                        logger.info("Lock acquired for streaming chat completion request.")
                        generate_common(inputs, generate_kwargs, streamer=generator_source)
                except Exception as e:
                    logger.error("Exception during generate", e)
                    # Just eat it since we're in another thread

            generate_thread_pool.submit(generate_stream)

            # AFAIK, if the passed generator is not an AsyncGenerator, it will
            # be iterated in a separate thread automatically.
            return StreamingResponse(
                stream_tokens_sse(generator_source),
                media_type="text/event-stream"
            )

        else:
            logger.info("Non-streaming chat completion request started.")
            # Non-streaming response

            # Push generation to another thread
            def generate_nonstream():
                try:
                    with lock: # NB This is a threading.Lock now, so it blocks
                        logger.info("Lock acquired for non-streaming chat completion request.")
                        return generate_common(inputs, generate_kwargs)
                except Exception as e:
                    logger.error("Exception during generate", e)
                    # Just eat it since we're in another thread

            generated_tokens = await asyncio.to_thread(generate_nonstream)

            # Decode the tokens
            generated_len = len(generated_tokens)
            text = processor.decode(generated_tokens, skip_special_tokens=True)
            logger.debug(f"Generated chat text: {text}")

            # Create the response
            response = ChatCompletionResponse(
                id=generate_id(),
                object="chat.completion",
                created=current_timestamp(),
                model=used_model or 'unknown',
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionMessage(role="assistant", content=text),
                        index=0,
                        finish_reason="length" if request.max_tokens else "stop"
                    )
                ],
                usage={
                    "prompt_tokens": prompt_len,
                    "completion_tokens": generated_len,
                    "total_tokens": prompt_len + generated_len,
                }
            )
            logger.info("Chat completion request processing completed.")
            return response

    except Exception as e:
        logger.error("Error during chat completion processing:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Exiting /v1/chat/completions endpoint.")


# New Endpoint: /v1/models
@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    logger.info("Models request received.")
    try:
        if model is None or model_name_loaded is None or model_loaded_time is None:
            logger.error("Model is not loaded.")
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        model_info = ModelInfo(
            id=model_name_loaded,
            created=model_loaded_time,
            owned_by="user",  # Adjust as needed
            permission=[],    # Can be populated with actual permissions if available
            root=model_name_loaded,
            parent=None
        )

        response = ModelsResponse(
            data=[model_info]
        )

        logger.info("Models response prepared successfully.")
        return response

    except Exception as e:
        logger.error("Error during models processing:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Exiting /v1/models endpoint.")


# Main function to parse arguments and start Uvicorn
def main():
    parser = argparse.ArgumentParser(description="Launch the Llama Vision OpenAI-Compatible API server.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory or HuggingFace model ID (e.g., 'gpt2')."
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit precision (requires appropriate support)."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model in 8-bit precision (requires appropriate support)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server to."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_max_tokens,
        help="Default maximum number of tokens to generate."
    )

    args = parser.parse_args()

    # Set environment variables based on parsed arguments
    os.environ["MODEL_NAME"] = args.model
    os.environ["LOAD_IN_4BIT"] = str(args.load_in_4bit)
    os.environ["LOAD_IN_8BIT"] = str(args.load_in_8bit)
    os.environ["MAX_TOKENS"] = str(args.max_tokens)

    # Run the app using Uvicorn with single worker and single thread
    uvicorn.run(
        "run_api:app",  # Ensure this matches the filename if different
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",  # Set to DEBUG for more detailed logs
        timeout_keep_alive=600,  # 10 minutes
        workers=1,  # Single worker to enforce global lock
        loop="asyncio",  # Ensure using asyncio loop
    )


if __name__ == "__main__":
    main()
