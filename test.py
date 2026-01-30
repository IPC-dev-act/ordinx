
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = "models/qwen2.5-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="cpu",          # CPU just to verify loading
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("✅ Qwen2.5-7B-Instruct loaded successfully")
EOF
