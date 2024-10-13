from transformers import MllamaForConditionalGeneration, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator

model_id = 'Llama-3.2-11B-Vision-Instruct'

q_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=[
        # More of a "feel good" thing to skip quantizing embedding
        # & lm_head layers.
        'vision_model.patch_embedding',
        'vision_model.gated_positional_embedding',
        'vision_model.gated_positional_embedding.tile_embedding',
        'vision_model.pre_tile_positional_embedding',
        'vision_model.pre_tile_positional_embedding.embedding',
        'vision_model.post_tile_positional_embedding',
        'vision_model.post_tile_positional_embedding.embedding',
        'language_model.model.embed_tokens',
        'language_model.lm_head',
        # Quantizing the following leads to CUDA assertion errors during
        # inference, so skip it.
        'multi_modal_projector'
    ]
)

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # NB LLM.int8() wants fp16 (it will warn otherwise)
    device_map='auto',
    quantization_config=q_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dest = 'Llama-3.2-11B-Vision-Instruct-int8'

# AFAIK, the model needs to reside on a single device before you save it
# However, moving it gets you a complaint from the accelerate module
# So I do it in 2 steps here... First the config:
model.config.save_pretrained(dest)

# Then the model (I assume accelerate knows how to handle multi-device models)
accelerator = Accelerator()
accelerator.save_model(model, dest)

# We'll just copy the preprocessor & tokenizer files manually. Is there a better way to do it?
# PS Don't forget chat_template.json
