import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./qlora-summariser/final_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_model_for_inference():
    print("Loading tokenizer from adapter directory...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    print("Attaching trained LoRA adapter")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return model, tokenizer


def summarize(model, tokenizer, dialogue: str, max_new_tokens: int = 100) -> str:
    prompt = (
        f"Instruction:\nSummarize the following conversation.\n\n"
        f"conversation:\n{dialogue}\n\n"
        f"Summary:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][prompt_length:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return summary.strip()


if __name__ == "__main__":
    model, tokenizer = load_model_for_inference()

    test_dialogues = [
        "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow :-)",
        "Tom: Are we still on for the movie tonight?\nSarah: Yes! 7pm right?\nTom: Yep, meet you outside the cinema.\nSarah: Sounds good, see you then!",
    ]

    for i, dialogue in enumerate(test_dialogues):
        print(f"\n{'=' * 60}")
        print(f"Example {i + 1}")
        print(f"{'=' * 60}")
        print(f"DIALOGUE:\n{dialogue}")
        summary = summarize(model, tokenizer, dialogue)
        print(f"\nGENERATED SUMMARY:\n{summary}")
