import argparse
import json
import torch
import os
from pathlib import Path
from utils.data import load_data, get_prompt
from utils.learning import get_model_and_tokenizer
from datasets import Dataset
from tqdm import tqdm
from transformers import logging as hf_logging


hf_logging.set_verbosity_error()


def main(
        data_dir,
        experiment,
        output_dir,
        input_content,
        checkpoint_foldername='checkpoints',
):

    ############################################
    #           Load and split data            #
    ############################################

    data_filepath = os.path.join(data_dir, 'data.json')
    split_filepath = os.path.join(data_dir, 'data_split')
    train_data, test_data = load_data(data_filepath, split_filepath)
    print(train_data[0])

    ############################################
    #                 Training                 #
    ############################################

    # Load the model
    tokenizer, model = get_model_and_tokenizer(experiment)

    # Construct the training prompts and test prompts
    train_samples = Dataset.from_dict({
        'text': [get_prompt(instance, input_content, 'train') for instance in train_data]
    })
    test_samples = Dataset.from_dict({
        'text': [get_prompt(instance, input_content, 'test') for instance in test_data],
        'label': [instance['label'] for instance in test_data]
    })
    #
    # if '+' in experiment:  # few-shot learning
    #
    #     # Define the training arguments
    #     checkpoint_folder = os.path.join(output_dir, checkpoint_foldername)
    #     checkpoint_subfolder = os.path.join(checkpoint_folder, f"{experiment}_checkpoints")
    #     Path(checkpoint_subfolder).mkdir(parents=True, exist_ok=True)



    ############################################
    #                Inference                 #
    ############################################



    # # save the predictions and references
    # response_folder = os.path.join(output_dir, response_foldername)
    # Path(response_folder).mkdir(parents=True, exist_ok=True)
    # output_filename = f"{experiment}_{input_content}_response"
    # output_filepath = os.path.join(response_folder, output_filename)
    # output = {
    #     'predictions': predictions,
    #     'labels': labels,
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
