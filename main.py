import argparse
import json
import torch
import os
from pathlib import Path
from utils.data import load_data, get_prompt
from utils.learning import get_train_args, get_peft_config
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer
from tqdm import tqdm
from transformers import logging as hf_logging


hf_logging.set_verbosity_error()


def main(
        data_dir,
        experiment,
        output_dir,
        input_content,
        checkpoint_foldername='checkpoints',
        adapter_foldername='saved_adapters',
        loss_foldername='loss'
):

    ############################################
    #           Load and split data            #
    ############################################

    data_filepath = os.path.join(data_dir, 'data.json')
    split_filepath = os.path.join(data_dir, 'data_split')
    train_data, test_data = load_data(data_filepath, split_filepath)

    ############################################
    #                 Training                 #
    ############################################

    # Step 1: Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_8bit=True,  # if we have enough GPU memory, we can set load_in_8bit as False
        device_map="auto"
    )

    # Step 2: Construct the training prompts and test prompts
    train_prompts = Dataset.from_dict({
        'text': [get_prompt(instance, input_content, 'train') for instance in train_data]
    })
    test_prompts = Dataset.from_dict({
        'text': [get_prompt(instance, input_content, 'train') for instance in test_data]
    })

    # Step 3: Define the training arguments
    checkpoint_folder = os.path.join(output_dir, checkpoint_foldername)
    checkpoint_subfolder = os.path.join(checkpoint_folder, f"{experiment}_checkpoints")
    Path(checkpoint_subfolder).mkdir(parents=True, exist_ok=True)
    training_args = get_train_args(checkpoint_subfolder)

    # Step 4: Use PEFT to only finetune part of its parameters
    peft_config = get_peft_config()

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_prompts,
        dataset_text_field="text",
        max_seq_length=4096,
        peft_config=peft_config,
    )

    # Start fine-tuning!
    trainer.train()

    # save the fine-tuned adapter
    adapter_folder = os.path.join(output_dir, adapter_foldername)
    Path(adapter_folder).mkdir(parents=True, exist_ok=True)
    adapter_path = os.path.join(adapter_folder, f"{experiment}_adapter")
    trainer.save_model(adapter_path)

    # save the loss of each step
    loss_folder = os.path.join(output_dir, loss_foldername)
    Path(loss_folder).mkdir(parents=True, exist_ok=True)
    loss_log_filename = f"loss_{experiment}"
    loss_log_filepath = os.path.join(loss_folder, loss_log_filename)
    with open(loss_log_filepath, 'w') as file:
        json.dump(trainer.state.log_history, file)

    ############################################
    #                Inference                 #
    ############################################

    # batch_size = 16
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_size="left")
    # tokenizer.pad_token = tokenizer.eos_token
    # pbar = tqdm(total=len(test_examples))
    # predictions, references = [], []
    # with torch.inference_mode():
    #     for start in range(0, len(test_examples), batch_size):
    #         end = min(start + batch_size, len(test_examples))
    #         texts = [sample for sample in test_dataset['text'][start: end]]
    #         input_ids = tokenizer(texts, return_tensors='pt', padding=True).to('cuda')
    #         output_tokens = model.generate(**input_ids, max_new_tokens=50, do_sample=False, use_cache=True)
    #         for ele in output_tokens:
    #             decoded_output = tokenizer.decode(ele, skip_special_tokens=True)
    #             predictions.append(decoded_output)
    #         for text, label in zip(test_dataset['text'][start: end], test_dataset['label'][start: end]):
    #             reference = text + label
    #             references.append(reference)
    #         pbar.update(end - start)
    # pbar.close()
    #
    # # save the predictions and references
    # response_folder = 'responses'
    # Path(response_folder).mkdir(parents=True, exist_ok=True)
    # output_filename = f"{experiment}_response"
    # output_filepath = os.path.join(response_folder, output_filename)
    # output = {
    #     'predictions': predictions,
    #     'references': references
    # }
    # with open(output_filepath, 'w') as file:
    #     json.dump(output, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # set up the arguments we could use
    parser.add_argument('-data_dir', '--data_dir',
                        help='The path to the annotation file.')
    parser.add_argument('-experiment', '--experiment',
                        help='The name of the experiment you want to run.')
    parser.add_argument('-input_content', '--input_content',
                        help='Which part of the Twitter stream you want to use')
    parser.add_argument('-output_dir', '--output_dir',
                        help='The name of the output directory.')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        experiment=args.experiment,
        output_dir=args.output_dir,
        input_content=args.input_content,
    )
