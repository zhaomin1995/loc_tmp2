from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments
)
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


def get_peft_config(peft_name='lora'):
    if peft_name == 'lora':
        peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=8, task_type="CAUSAL_LM")
    elif peft_name == 'prefix-tuning':
        pass  # this is on the research agenda
    elif peft_name == 'prompt-tuning':
        pass  # this is on the research agenda
    else:
        raise ValueError("Please check the name of peft method")
    return peft_config


def get_model_and_tokenizer(experiment):
    if experiment == 'flan_ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'google/flan-ul2',
            load_in_8bit=True,  # if we have enough GPU memory, we can set load_in_8bit as False
            device_map="auto"
        )
    elif experiment == 'flan_t5':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl",
            device_map="auto",
            load_in_8bit=True
        )
    elif experiment == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/ul2")
    elif experiment == 't5':
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    # elif experiment == 'ul2':
    #     model_name = 'google/ul2'
    # elif experiment == 'ul2':
    #     model_name = 'google/ul2'
    # elif experiment == 'ul2':
    #     model_name = 'google/ul2'
    else:
        raise ValueError("Please check the name of the experiment")
    return tokenizer, model
