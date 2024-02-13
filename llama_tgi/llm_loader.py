from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from myapp.settings import DEVICE

import torch

tokenizer = AutoTokenizer.from_pretrained("AbdulHannanMujawar/llama-fine-tuned")
model = AutoModelForCausalLM.from_pretrained("AbdulHannanMujawar/llama-fine-tuned")

def generate(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def summarize(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
    
input_str = """
### Instruction Below is a conversation between a human and an AI agent. Write a summary of the conversation.\n\n### Input:\nuser: looking to change my flight Friday, Oct 27. GRMSKV to DL4728 from SLC to ORD. Is that an option and what is the cost? Jess\nagent: The difference in fare is $185.30. This would include all airport taxes and fees. The ticket is non-refundable changeable with a fee, *ALS and may result in additional fare collection for changes when making a future changes. *ALS\n\n### Response:\nCustomer is looking to change the flight on Friday Oct 27 is that an option and asking about cost. Agent replying that there is an difference in fare and this would include all airport taxes and fees and ticket is non refundable changeable with a fee.
"""
summarize(input_str)