import json
import re


EXEMPLARS = [
    (
        "Read the tweets and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "Please only select the number listed below.\n\n"
        "One thing that has surprised me since moving to Dallas is how beautiful the Texas sky can be.\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Dallas when the tweet was published.\n"
        "Answer: 1."
    ),
    (
        "Read the tweets and determine if the author of the tweet is located in Dallas when the tweet was published. "
        "Please only select the number listed below.\n\n"
        "breaking news: the seattle kraken are being removed from the nhl because the booktok fans are done with them. "
        "rip seattle kraken 2021-2023\n\n"
        "OPTIONS:\n"
        "1. Yes.\n"
        "2. I cannot determine if the author of the tweet is located in Seattle when the tweet was published.\n"
        "Answer: 2."
    )
]


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = text.strip()
    return text


def load_data(data_filepath, split_filepath):
    train_data, test_data = [], []

    with open(split_filepath, 'r') as file:
        splits = json.load(file)
        train_ids = splits['train']
        test_ids = splits['test']

    with open(data_filepath, 'r') as file:
        for line in file:
            if len(line) == 0:
                continue
            item = json.loads(line)
            kept_annotations = [item[key] for key in item.keys() if key.startswith("Answer.Q1_")]
            if len(kept_annotations) == 0:
                continue
            texts = [
                clean_text(item['context8_tweettext']),
                clean_text(item['context9_tweettext']),
                clean_text(item['context10_tweettext']),
                clean_text(item['context11_tweettext']),
                clean_text(item['context12_tweettext']),
                clean_text(item['context13_tweettext']),
            ]
            instance = {
                'texts': texts,
                'label': item['adjudicated_label'],
                'location': item['anchor_location']
            }
            if item['instance_id'] in train_ids:
                train_data.append(instance)
            if item['instance_id'] in test_ids:
                test_data.append(instance)

    return train_data, test_data


def combine_texts(texts, input_content):
    if input_content == 'target':
        return texts[3]
    elif input_content == 'early_target':
        return '\n'.join(texts[:4])
    elif input_content == 'target_later':
        return '\n'.join(texts[3:])
    elif input_content == 'all':
        return '\n'.join(texts)
    else:
        raise ValueError("Please check the input content")


def format_prompt(prompt, label, data_type, exemplar):

    instruction = '\n\n'.join(EXEMPLARS)
    model_input = ""
    if data_type == 'train':
        if exemplar == 'few-shot':
            model_input += f"### Instruction: {instruction}\n"
        model_input += f"### Prompt: {prompt}{label}"
    else:
        if exemplar == 'few-shot':
            model_input += f"### Instruction: {instruction}\n"
        model_input += f"### Prompt: {prompt}"

    return model_input


def get_prompt(instance, input_content, data_type, exemplar):

    tweet_text = combine_texts(instance['texts'], input_content)
    label = instance['label']
    location = instance['location']
    prompt = (
        f"Read the tweets and determine if the author of the tweet is located at {location} when the tweet was published. "
        "Please only select the number listed below.\n\n"
        f"{tweet_text}\n\n"
        f"OPTIONS:\n"
        f"1. Yes.\n"
        f"2. I cannot determine if the author of the tweet is located at {location} when the tweet was published.\n"
        "ANSWER: "
    )
    prompt = format_prompt(prompt, label, data_type, exemplar)

    return prompt


