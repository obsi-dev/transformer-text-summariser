import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "knkarthick/samsum"
OUTPUT_DIR = "./qlora-summariser"
MAX_SEQ_LEN = 768

TRAIN_SUBSET_SIZE = 400
EVAL_SUBSET_SIZE = 50

# quantization config

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# tokenizer + base model

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)

print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# LoRA config

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# formatting and such

print(f"Loading dataset: {DATASET_NAME}")
raw_dataset = load_dataset(DATASET_NAME)


def format_example(example):
    # New trl API wants separate "prompt" and "completion" columns
    # instead of one glued-together string + a marker to search for.
    prompt = (
        f"### Instruction:\nSummarize the following conversation.\n\n"
        f"### Conversation:\n{example['dialogue']}\n\n"
        f"### Summary:\n"
    )
    completion = example["summary"] + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


print("Formatting dataset")
train_dataset = raw_dataset["train"].map(
    format_example, remove_columns=raw_dataset["train"].column_names
)
eval_dataset = raw_dataset["validation"].map(
    format_example, remove_columns=raw_dataset["validation"].column_names
)

train_dataset = train_dataset.select(range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))
eval_dataset = eval_dataset.select(range(min(EVAL_SUBSET_SIZE, len(eval_dataset))))

print(
    f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} samples."
)

# training config
# completion_only_loss=True is the new replacement for
# DataCollatorForCompletionOnlyLM — since our dataset has separate
# "prompt" and "completion" columns, trl already knows exactly where
# the completion starts, no marker-string search needed.

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    max_length=MAX_SEQ_LEN,
    completion_only_loss=True,
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

if __name__ == "__main__":
    print("Starting training...\n")
    trainer.train()

    save_path = f"{OUTPUT_DIR}/final_adapter"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Done. Adapter saved to {save_path}")
