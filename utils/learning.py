from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig


def get_train_args(output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="epoch",
        save_steps=50,
        save_total_limit=5,
        num_train_epochs=6,
    )
    return training_args


def get_peft_config():
    peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=8, task_type="CAUSAL_LM")
    return peft_config

