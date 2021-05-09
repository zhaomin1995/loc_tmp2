import json
import torch
import transformers as ppb
import numpy as np
from collections import Counter


def load_data(data_dir):
    """
    load the corpus
    :param data_dir: path to the annotation file
    :return: instances (dictionary)
    """
    instances = []
    with open(data_dir, 'r') as datafile:
        lines = datafile.read().split("\n")[:-1]
        for line in lines:
            instance = json.loads(line)
            # skip the instances without annotations after filtering out
            num_workers = len([key for key in instance.keys() if key.startswith("Answer.Q1")])
            if num_workers == 0:
                continue
            else:
                instances.append(instance)
    return instances


def add_baseline_output(instances):
    """
    add majority baseline output to the instance (dictionary)
    :param instances: instances without majority baseline output
    :return: instances with majority baseline output
    """
    all_labels = [x['adjudicated_label'] for x in instances]
    label_count = Counter(all_labels)
    majority_label = label_count.most_common()[0][0]
    for instance in instances:
        instance['baseline_output'] = majority_label
    return instances


def add_bert_output(instances, anchor_only):
    """
    add bert output to the instances
    :param instances: instances without bert output
    :param anchor_only: only extract feature of anchor tweet or anchor+context
    :return:
    """
    # load the BERT model and the tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_class, bert_tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    bert_model = bert_model_class.from_pretrained(pretrained_weights).to(device)
    tokenizer = bert_tokenizer_class.from_pretrained(pretrained_weights)

    if anchor_only:

        # tokenize the tweet text
        text_tokenized = [tokenizer.encode(x['anchor_tweettext'], add_special_tokens=True) for x in instances]

        # pad the sentences and prepare the mask for using the ids
        max_len = max([len(x) for x in text_tokenized])
        all_padded = np.array([i + [0] * (max_len - len(i)) for i in text_tokenized])
        all_attention_mask = np.where(all_padded != 0, 1, 0)

        # convert the id and mask into tensor
        all_padded_ids = torch.tensor(all_padded)
        all_attention_mask = torch.tensor(all_attention_mask)

        # add the bert input into the passed argument -- instances
        for index, instance in enumerate(instances):
            with torch.no_grad():
                small_padded_ids = all_padded_ids[index].unsqueeze(0).to(device)
                small_attention_mask = all_attention_mask[index].unsqueeze(0).to(device)
                last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

            # after getting the representation, move the tensor to CPU to free GPU memory
            anchor_bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
            instance['anchor_bertoutput'] = anchor_bert_feature

    else:

        # tokenize the tweet text
        text_tokenized = []
        for instance in instances:
            text_tokenized.append(tokenizer.encode(instance['anchor_tweettext'], add_special_tokens=True))
            for i in range(8, 14):
                text_tokenized.append(tokenizer.encode(instance[f"context{i}_tweettext"], add_special_tokens=True))

        # pad the sentences and prepare the mask for using the ids
        max_len = max([len(x) for x in text_tokenized])
        all_padded = np.array([i + [0] * (max_len - len(i)) for i in text_tokenized])
        all_attention_mask = np.where(all_padded != 0, 1, 0)

        # convert the id and mask into tensor
        all_padded_ids = torch.tensor(all_padded)
        all_attention_mask = torch.tensor(all_attention_mask)

        # add bert output for each of seven tweets of every instance
        for index, instance in enumerate(instances):

            # add bert output of anchor tweet
            with torch.no_grad():
                small_padded_ids = all_padded_ids[index * 7].unsqueeze(0).to(device)
                small_attention_mask = all_attention_mask[index * 7].unsqueeze(0).to(device)
                last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

            # move the tensor to CPU to free GPU memory
            anchor_bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
            instance['anchor_bertoutput'] = anchor_bert_feature

            # add bert output of context tweets
            for i in range(8, 14):
                with torch.no_grad():
                    small_padded_ids = all_padded_ids[index * 7 + (i - 7)].unsqueeze(0).to(device)
                    small_attention_mask = all_attention_mask[index * 7 + (i - 7)].unsqueeze(0).to(device)
                    last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

                # move the tensor to CPU to free GPU memory
                context_bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
                instance[f'context{i}_bertoutput'] = context_bert_feature

    return instances


