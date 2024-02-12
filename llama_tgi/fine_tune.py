import json
import re

from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from trl import SFTTrainer
from myapp.settings import HF_TOKEN

login(HF_TOKEN)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


dataset = load_dataset("Salesforce/dialogstudio", "TweetSumm")

DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()

def generate_training_prompt(
    conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
  return f"""### Instruction {system_prompt}

### Input:
{conversation.strip()}

### Response:
{summary}
""".strip()

def clean_text(text):
  text = re.sub(r"http\S+","",text)
  text = re.sub(r"@[^\s]+","",text)
  text = re.sub(r"\s+"," ",text)
  return re.sub(r"\^[^ ]+","",text)

def create_conversation_text(data_point):
  text = ""
  for item in data_point["log"]:
    user = clean_text(item["user utterance"])
    text += f"user: {user.strip()}\n"
    agent = clean_text(item["system response"])
    text += f"agent: {agent.strip()}\n"

    return text
  
def generate_text(data_point):
  summaries= json.loads(data_point["original dialog info"])["summaries"][
      "abstractive_summaries"
  ]
  summary = summaries[0]
  summary = " ".join(summary)

  conversation_text = create_conversation_text(data_point)
  return {
      "conversation": conversation_text,
      "summary": summary,
      "text": generate_training_prompt(conversation_text, summary),
  }

def process_dataset(data: Dataset):
  return (
      data.shuffle(seed=42)
      .map(generate_text)
      .remove_columns([
          "original dialog id",
          "new dialog id",
          "dialog index",
          "original dialog info",
          "log",
          "prompt"
      ])
  )

dataset["train"] = process_dataset(dataset["train"])
dataset["validation"] = process_dataset(dataset["validation"])
dataset["test"] = process_dataset(dataset["test"])

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

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

model.config.quantization_config.to_dict()

lora_alpha = 32
lora_dropout = 0.05
lora_r = 16

peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM"
)

OUTPUT_DIR = "experiments"

training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,

)

trainer.train()
trainer.save_model()

from peft import AutoPeftModelForCausalLM

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,

)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")