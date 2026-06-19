import torch 
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

model_name = "meta-llama/Meta-Llama-3-8B"
dataset_path = "train.jsonl"
output_dir = "./qora_finetuned"

tokenizer = AutoTokenizer.pretrained_mode(model_name, use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

# 4-BIT QUANTIZATION (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True
)

# model
model = AutoModelForCausalLM(
    model_name,
    quantization_config = bnb_config,
    device_map = "auto"
)

# lora config
lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj"]   
)

def format_chat(text):
    text = tokenizer.apply_template(
        example["message"],
        tokenizer = False,
        add_generation_template = False
    )
    return {"text": text}

dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset.map(format_chat, remove_columns = dataset["train"].column_names)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"], 
    peft_config=lora_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_arguments,
    max_seq_length=2048,
    peft_config=lora_config
)

trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training completed and model saved at:", output_dir)









