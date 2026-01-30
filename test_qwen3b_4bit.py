import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "models/qwen2.5-3b-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

def extract_json(text: str) -> dict:
    """
    Extract the LAST valid JSON object from model output.
    (LLMs often echo JSON templates in the prompt.)
    """
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)

    parsed = []
    for c in candidates:
        try:
            parsed.append(json.loads(c))
        except json.JSONDecodeError:
            continue

    if not parsed:
        raise ValueError("No valid JSON object found in output:\n" + text)

    return parsed[-1]  # <-- key change: take last JSON

messages = [
    {"role": "system", "content": (
        "You are a data extraction engine. Output ONLY valid JSON. "
        "No markdown. No explanation."
    )},
    {"role": "user", "content": (
        "Extract the total amount and currency from the text below.\n"
        "Rules:\n"
        "- total: keep decimal comma if present\n"
        "- currency: 3-letter code if present (EUR, USD, etc)\n"
        "- If missing, use null\n\n"
        "TEXT: Total: 14,00 EUR\n\n"
        "Return JSON with exactly these keys:\n"
        "{\"total\": null, \"currency\": null}"
    )},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

decoded = tokenizer.decode(out[0], skip_special_tokens=True)

# Some templates echo the prompt; grab the tail JSON safely
data = extract_json(decoded)
print(json.dumps(data, ensure_ascii=False))
