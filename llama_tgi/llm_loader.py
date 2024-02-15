from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from myapp.settings import DEVICE, model, tokenizer

from peft import LoraConfig, PeftModel
import torch


MODEL_NAME="meta-llama/Llama-2-7b-hf"
NEW_MODEL_NAME="AbdulHannanMujawar/experiments"

#tokenizer = AutoTokenizer.from_pretrained("AbdulHannanMujawar/llama-fine-tuned")
#model = AutoModelForCausalLM.from_pretrained("AbdulHannanMujawar/llama-fine-tuned")

model = PeftModel.from_pretrained(model, NEW_MODEL_NAME)


# def create_model_and_tokenizer():
#   bnb_config = BitsAndBytesConfig(
#       load_in_4bit=True,
#       bnb_4bit_quant_type="nf4",
#       bn_4bit_compute_dtype=torch.float16,
#   )

#   model = AutoModelForCausalLM.from_pretrained(
#       MODEL_NAME,
#       use_safetensors=True,
#       quantization_config=bnb_config,
#       trust_remote_code=True,
#       device_map="auto"
#   )

#   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#   tokenizer.pad_token = tokenizer.eos_token
#   tokenizer.padding_side = "right"

#   return model, tokenizer



def generate(prompt):
    #input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    #output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #return generated_text
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    output = []
    for seq in sequences:
        output.append(seq['generated_text'])
        print(f"Result: {seq['generated_text']}")

def summarize(text: str):
    innput_text = input_str = f"""
    ### Instruction Below is a conversation between a human and an AI agent. Write a summary of the conversation.\n\n### Input:{text}### Response:
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.to(DEVICE).generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


input_str = """
### Instruction Below is a conversation between a human and an AI agent. Write a summary of the conversation.\n\n### Input:\nuser: looking to change my flight Friday, Oct 27. GRMSKV to DL4728 from SLC to ORD. Is that an option and what is the cost? Jess\nagent: The difference in fare is $185.30. This would include all airport taxes and fees. The ticket is non-refundable changeable with a fee, *ALS and may result in additional fare collection for changes when making a future changes. *ALS\### Response:\nCustomer is looking to change the flight on Friday Oct 27 is that an option and asking about cost. Agent replying that there is an difference in fare and this would include all airport taxes and fees and ticket is non refundable changeable with a fee.
"""
#summarize(input_str)