from collections import OrderedDict
from pprint import pprint

import torch
from transformers import MllamaConfig, MllamaForConditionalGeneration
from accelerate import init_empty_weights, infer_auto_device_map
from torch.nn import ModuleList


model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

config = MllamaConfig.from_pretrained(model_id)

with init_empty_weights():
    model = MllamaForConditionalGeneration(config)

def gather_named_children(mdl, result: list[str], parent: list[str]|None=None, recurse: bool=True):
    if parent is None:
        parent = []

    for n, m in mdl.named_children():
        names = list(parent) # copy
        names.append(n)

        is_list = isinstance(m, ModuleList)
        has_params = len(list(m.parameters(recurse=False))) > 0
        has_buffers = len(list(m.buffers(recurse=False))) > 0

        if has_params or has_buffers or not recurse:
            #print(f"{'.'.join(names)}")
            result.append('.'.join(names))

        if recurse:
            gather_named_children(m, result, parent=names, recurse=not is_list)

result: list[str] = []
gather_named_children(model, result)

device_map = OrderedDict([
    (n, 0) for n in result
])

print('device_map = ', end='')
pprint(device_map)
print()

out = []
for n, m in model.named_modules():
    if (('embed' in n) or
        ('lm_head' in n) or
        ('multi' in n)):
        out.append(n)

print('llm_int8_skip_modules = ', end='')
pprint(out)
