import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
output_dir = "/home/raza/EMRV1/EMRV2/Baseline/medinotes/llama70b_4bit_prepared"

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(model_name)

print("Loading model in 4-bit on a single GPU…")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_cfg,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

print("Saving prepared model…")
model.save_pretrained(output_dir)
tok.save_pretrained(output_dir)

print("Done.")
