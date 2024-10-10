# Originally from https://github.com/sam-paech/antislop-sampler @ e931054 || Apache-2.0
# Modified for Llama 3.2 Vision by Allan Saddi <allan@saddi.com>
import argparse
import asyncio
import json
import queue  # Import queue for thread-safe communication
import os
import time
from typing import List, Dict, Optional, Any, AsyncGenerator
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
    PreTrainedModel,
    PreTrainedTokenizer,
    MllamaForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Vision OpenAI-Compatible API")

# Global variables to hold the model and tokenizer
model: Optional[PreTrainedModel] = None
processor: Optional[PreTrainedTokenizer] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store model metadata
model_loaded_time: Optional[int] = None
model_name_loaded: Optional[str] = None

# Define a global asyncio.Lock to enforce single concurrency
lock = asyncio.Lock()

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
    temperature: Optional[float] = Field(default=1.0, ge=0.0, description="Sampling temperature")
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


# Utility functions

# Startup event to load model and tokenizer
@app.on_event("startup")
async def load_model_and_tokenizer():
    global processor, model
    global model_loaded_time, model_name_loaded

    model_name = os.environ['MODEL_NAME'] # FIXME This really comes from the environment??

    # Load tokenizer
    logger.info(f"Loading tokenizer for model '{model_name}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        logger.info("Tokenizer loaded.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise e

    # Load model with appropriate precision
    logger.info(f"Loading model '{model_name}'...")
    try:
        model = MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e  # Let FastAPI handle the startup failure

    logger.info("Model and tokenizer loaded successfully.")

    # Store model metadata
    model_loaded_time = current_timestamp()
    model_name_loaded = model_name


def generate_id() -> str:
    return str(uuid.uuid4())


def current_timestamp() -> int:
    return int(time.time())


# Utility function for streaming responses

async def stream_tokens_sync(generator: Any, is_chat: bool = False) -> AsyncGenerator[str, None]:
    """
    Converts a synchronous generator to an asynchronous generator for streaming responses.
    Formats the output to match OpenAI's streaming response format.
    """
    q = queue.Queue()

    def generator_thread():
        try:
            logger.debug("Generator thread started.")
            for token in generator:
                q.put(token)
                logger.debug(f"Token put into queue: {token}")
            q.put(None)  # Signal completion
            logger.debug("Generator thread completed.")
        except Exception as e:
            logger.error(f"Exception in generator_thread: {e}")
            q.put(e)  # Signal exception

    # Start the generator in a separate daemon thread
    thread = threading.Thread(target=generator_thread, daemon=True)
    thread.start()
    logger.debug("Generator thread initiated.")

    try:
        while True:
            token = await asyncio.to_thread(q.get)
            logger.debug(f"Token retrieved from queue: {token}")

            if token is None:
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

            if isinstance(token, Exception):
                # Handle exceptions by sending a finish_reason with 'error'
                error_data = {
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": "error"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                logger.error(f"Exception during token streaming: {token}")
                break  # Exit the loop after handling the error

            # Since our generator is a TextIteratorStreamer:
            # 1. The "tokens" we recieve are already decoded
            # 2. We can tell it to skip over the prompt
            #
            # So we can just emit the newly-received token as-is here

            # Prepare the data in OpenAI's streaming format
            data = {
                "choices": [
                    {
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

            # Yield the formatted data as a Server-Sent Event (SSE)
            yield f"data: {json.dumps(data)}\n\n"
            logger.debug("Yielded token to client.")

            # Yield control back to the event loop
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.warning("Streaming task was cancelled by the client.")
    except Exception as e:
        logger.error(f"Unexpected error in stream_tokens_sync: {e}")
    finally:
        logger.debug("Exiting stream_tokens_sync.")


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


# Endpoint: /v1/chat/completions
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, req: Request):
    logger.info("Chat completion request received, waiting for processing...")
    try:
        if model is None or processor is None:
            logger.error("Model and tokenizer are not loaded.")
            raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded.")

        # Use the model specified in the request or default
        used_model = request.model if request.model else model_name_loaded

        # Build the prompt from chat messages
        images = await extract_images(request.messages)
        prompt = processor.apply_chat_template(request.messages, add_generation_prompt=True)
        logger.debug(f"Constructed prompt from messages: {prompt}")

        img_list = None if len(images) == 0 else images # MllamaProcessor does not like empty image lists. Use None instead.
        inputs = processor(img_list, prompt, return_tensors='pt').to(model.device)

        if request.stream:
            logger.info("Streaming chat completion request started.")
            # Streaming response
            generator_source = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True) # Why isn't this documented outside of the source?!
            # TODO sampler settings?!
            generator_kwargs = dict(**inputs, max_new_tokens=(2048 if request.max_tokens is None else request.max_tokens), streamer=generator_source)
            # FIXME Non-streaming version holds lock around entire generate call
            # I feel like this should do the same...
            gen_thread = threading.Thread(target=model.generate, kwargs=generator_kwargs)
            gen_thread.start()

            async def streaming_generator():
                async with lock:
                    logger.info("Lock acquired for streaming chat completion request.")
                    try:
                        async for token in stream_tokens_sync(generator_source, is_chat=True):
                            yield token
                    except Exception as e:
                        logger.error(f"Exception in streaming_generator: {e}")
                    finally:
                        logger.info("Streaming generator completed and lock released.")

            return StreamingResponse(
                streaming_generator(),
                media_type="text/event-stream"
            )

        else:
            logger.info("Non-streaming chat completion request started.")
            async with lock:
                logger.info("Lock acquired for non-streaming chat completion request.")
                # Non-streaming response
                # TODO sampler settings???
                # FIXME Default max_new_tokens should be defined elsewhere
                output = model.generate(**inputs, max_new_tokens=(2048 if request.max_tokens is None else request.max_tokens))
                # output tensor includes the prompt, so skip over it
                prompt_len = inputs.input_ids.shape[-1]
                generated_ids = output[:, prompt_len:]
                # it's a batch of 1, just grab the main result
                generated_tokens = generated_ids[0]

            # Decode the tokens
            generated_len = len(generated_tokens)
            text = processor.decode(generated_tokens, skip_special_tokens=True)
            logger.debug(f"Generated chat text: {text}")

            # Create the response
            response = ChatCompletionResponse(
                id=generate_id(),
                object="chat.completion",
                created=current_timestamp(),
                model=used_model,
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
        logger.error(f"Error during chat completion processing: {e}")
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
        logger.error(f"Error during models processing: {e}")
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

    args = parser.parse_args()

    # Set environment variables based on parsed arguments
    os.environ["MODEL_NAME"] = args.model

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
