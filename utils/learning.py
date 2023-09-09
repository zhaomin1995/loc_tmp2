from transformers import (
    BertModel,
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
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="epoch",
        save_steps=50,
        save_total_limit=5,
        num_train_epochs=1,
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


def get_model_and_tokenizer(experiment, cache_dir):
    if '+' in experiment:
        experiment = experiment.replace("+", "")
    if experiment == 'flan_ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'google/flan-ul2',
            load_in_8bit=True,  # if we have enough GPU memory, we can set load_in_8bit as False
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'flan_t5':
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl",
            device_map="auto",
            load_in_8bit=True,
            cache_dir=cache_dir
        )
    elif experiment == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained("google/ul2")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/ul2",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 't5':
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5-v1_1-xxl",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'flan_alpaca':
        tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-xxl")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "declare-lab/flan-alpaca-xxl",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=cache_dir
        )
    elif experiment == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
    else:
        raise ValueError("Please check the name of the experiment")
    return tokenizer, model
