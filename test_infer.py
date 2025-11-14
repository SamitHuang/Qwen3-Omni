import os
import soundfile as sf
import torch
import numpy as np
import time
import random

from transformers import AutoConfig, AutoModel
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen3OmniMoeConfig
from qwen_omni_utils import process_mm_info

debug = False

SEED = 42
# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if debug:
    # num hidden layers is changed to 1 in this model path
    MODEL_PATH = "/home/public/yx/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
else:    
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


model_load_start = time.time()
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    # low_cpu_mem_usage=True, # skip model init, just load weights
    dtype="bfloat16",  # "auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    # attn_implementation="sdpa",  # TODO:
)
model_load_end = time.time()
print(f"Model initialization took {model_load_end - model_load_start:.2f} seconds.")

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id
if pad_token_id is None:
    raise ValueError("Processor does not define pad or eos token ids.")
model.config.pad_token_id = pad_token_id
model.generation_config.pad_token_id = pad_token_id
if tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
if hasattr(model, "thinker"):
    model.thinker.generation_config.pad_token_id = pad_token_id
    if tokenizer.eos_token_id is not None:
        model.thinker.generation_config.eos_token_id = tokenizer.eos_token_id

conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            # {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},  # FIXME: audioread.ffdec.NotInstalledError 
            # {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
            {"type": "text", "text": "What is the capital of France? Answer in 10 words."}
        ],
    },
]

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text,
                   audio=audios,
                   images=images,
                   videos=videos,
                   return_tensors="pt",
                   padding=True,
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
if "attention_mask" not in inputs:
    mask = (inputs["input_ids"] != pad_token_id).long()
    inputs["attention_mask"] = mask

model_device = getattr(model, "device", None)
if model_device is None:
    model_device = next(model.parameters()).device
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        tensor_dtype = value.dtype
        if torch.is_floating_point(value):
            tensor_dtype = model.dtype
        inputs[key] = value.to(device=model_device, dtype=tensor_dtype)

# Inference: Generation of the output text and audio
generation_start = time.time()
text_ids, audio = model.generate(
    **inputs,
    speaker="Ethan",
    thinker_return_dict_in_generate=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
generation_end = time.time()
print(f"Generation took {generation_end - generation_start:.2f} seconds.")

if hasattr(text_ids, "sequences"):
    text_sequences = text_ids.sequences
else:
    text_sequences = text_ids

text = processor.batch_decode(
    text_sequences[:, inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(text)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
