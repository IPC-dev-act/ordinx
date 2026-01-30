import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "models/qwen2.5-7b-instruct"

# Helps with fragmentation on small VRAM GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Offload a bit to CPU if needed (saves VRAM at load time)
max_memory = {0: "7GiB", "cpu": "32GiB"}  # adjust cpu if needed

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("✅ Qwen loaded (4-bit)")

prompt = "Reply JSON only: {\"ok\": true}"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
