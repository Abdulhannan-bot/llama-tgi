from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
MODEL_NAME="meta-llama/Llama-2-7b-hf"
from peft import LoraConfig, PeftModel

# NEW_MODEL_NAME="AbdulHannanMujawar/experiments"

def create_model_and_tokenizer():
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bn_4bit_compute_dtype=torch.float16,
  )

  model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      use_safetensors=True,
      quantization_config=bnb_config,
      trust_remote_code=True,
      device_map="auto"
  )

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  return model, tokenizer