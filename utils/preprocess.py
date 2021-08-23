import os
import re
import json
import random
import spacy
import torch
import transformers as ppb
import numpy as np
import torchvision.models as models
from collections import Counter, defaultdict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer


def load_data(data_dir, mode):
    """
    load the corpus, and add task info
    :param data_dir: path to the annotation file
    :param mode: baseline / anchor_text_only / anchor_image_only / anchor_text_image / complicated_nn
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
                # add task info
                instance['tasktype'] = mode
                instances.append(instance)
    return instances


def load_mpqa(mpqa_path):
    """
    load MPQA lexicon
    :param mpqa_path: the path of MPQA lexicon file
    :return: a dictionary, {"strongsubj": [wordlist], "weaksubj": [wordlist]}
    """
    res = defaultdict(list)
    with open(mpqa_path, 'r') as file:
        lines = file.read().split("\n")[:-1]
        for line in lines:
            attributes = line.split(" ")
            res[attributes[0].split("=")[1]].append(attributes[2].split("=")[1])
    return res


def split_instances(instances, split_file='saved_split'):
    """
    split the instances using the saved split file
    :param instances: the instances need to be split into train, dev, and test
    :param split_file: the name of the saved split file
    :return: train_instances, dev_instances, and test_instances
    """
    # load the split file
    with open(split_file, 'r') as splitFile:
        split_dict = json.loads(splitFile.read())

    # load the anchor id of train, dev, and test
    train_ids = split_dict['train']
    dev_ids = split_dict['dev']
    test_ids = split_dict['test']

    # load the instances of train, dev, and test
    train_instances = []
    dev_instances = []
    test_instances = []
    for instance in instances:
        if instance['instance_id'] in train_ids:
            train_instances.append(instance)
        if instance['instance_id'] in dev_ids:
            dev_instances.append(instance)
        if instance['instance_id'] in test_ids:
            test_instances.append(instance)

    # shuffle the instances to improve the model training
    random.shuffle(train_instances)
    random.shuffle(dev_instances)
    random.shuffle(test_instances)

    return train_instances, dev_instances, test_instances


def feat_polished(instances):
    """
    split the integer feature values into some bins and convert it into string
    :param instances: instances with integer values
    :return: instances with string values
    """
    # get the statistics of the features whose value is integer
    feat_dicts = [value for instance in instances for key, value in instance.items() if key.endswith("addfeat")]
    keys = [key for key, value in feat_dicts[0].items() if isinstance(value, int)]
    feat_numbers = {key: [feat_dict[key] for feat_dict in feat_dicts] for key in keys}
    cutted_dict = defaultdict(dict)
    for key, value in feat_numbers.items():
        stats = Counter(value)
        # only get the most common value
        most_common = dict(stats.most_common(int(len(stats) / 6) + 1))
        for small_key in most_common.keys():
            cutted_dict[key].update({small_key: str(small_key)})
        cutted_dict[key].update({'other': 'other'})

    # update the feature value
    for instance in instances:
        keys = [key for key in instance.keys() if key.endswith("addfeat")]
        for key in keys:
            feat_values = instance[key]
            for feat_name in list(feat_values.keys()):
                feat_value = feat_values[feat_name]
                if not isinstance(feat_value, int):
                    continue
                if feat_value in cutted_dict.keys():
                    instance[key][feat_name] = cutted_dict[feat_name][feat_value]
                else:
                    instance[key][feat_name] = cutted_dict[feat_name]['other']

    return instances


def add_additional_features(instances, mpqa_lexicon):
    """
    extract additional features
    :param instances: instances without additional features
    :param mpqa_lexicon: the loaded mpqa lexicon (dictionary)
    :return: instances with additional features
    """
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('emoji', first=True)
    pbar = tqdm(total=len(instances))
    for instance in instances:
        # ensure the order is the same as in the later part
        keys = sorted([key for key in instance.keys() if key.endswith("tweettext")])
        location = instance['anchor_location']
        for key in keys:
            tweet = nlp(instance[key])
            featkey = key.split("_")[0] + "_addfeat"
            addfeat = {}

            # entirely uppercase words
            num_entireuppercasewords = len(['x' for token in tweet if token.text.isupper()])
            addfeat['num_entireuppercasewords'] = num_entireuppercasewords

            # the number of URLs
            num_urls = len(['x' for token in tweet if token.like_url])
            addfeat['num_urls'] = num_urls

            # the number of exclamation marks
            num_exclamationmarks = len(['x' for token in tweet if token.text == '!'])
            addfeat['num_exclamationmarks'] = num_exclamationmarks

            # the number of strongly subjective words in MPQA lexicon
            num_strongsubj = len([token for token in tweet if token.text in mpqa_lexicon['strongsubj']])
            addfeat['num_strongsubj'] = num_strongsubj

            # the number of weakly subjective words in MPQA lexicon
            num_weaksubj = len([token for token in tweet if token.text in mpqa_lexicon['weaksubj']])
            addfeat['num_weaksubj'] = num_weaksubj

            # the number of emoji
            num_emoji = len(tweet._.emoji)
            addfeat['num_emoji'] = num_emoji

            # the three most common emoji (in the form of description)
            emoji_desc_lists = [token._.emoji_desc for token in tweet if token._.is_emoji]
            emoji_count = Counter(emoji_desc_lists).most_common(3)
            for index, x in enumerate(emoji_count):
                addfeat[f"no.{index + 1}_emoji"] = x[0]

            # the number of tokens
            num_tokens = len(tweet)
            addfeat['num_tokens'] = num_tokens

            # the number of elongated words
            elong_pattern = re.compile("([a-zA-Z])\\1{2,}")
            num_elong = len(['x' for token in tweet if bool(elong_pattern.search(token.text))])
            addfeat['num_elong'] = num_elong

            # the number of hashtags
            num_hashtags = len(['x' for token in tweet if token.text.startswith("#")])
            addfeat['num_hashtags'] = num_hashtags

            # the number of first letter uppercased words
            num_uppercasewords = len(['x' for token in tweet if token.text[0].isupper()])
            addfeat['num_uppercasewords'] = num_uppercasewords

            # the surround words/lemma/pos/hashtag/reply
            contain_location = False
            for token in tweet:
                if location in token.text:
                    contain_location = True

                    # check if the location is included in a hashtag
                    addfeat['loc_hashtag'] = '1' if token.text.startswith("#") else '0'

                    # check if the location is included in a mention
                    addfeat['loc_mention'] = '1' if token.text.startswith("@") else '0'
                    break
            if not contain_location:
                addfeat['loc_hashtag'] = '0'
                addfeat['loc_mention'] = '0'

            instance[featkey] = addfeat
        pbar.update(1)
    pbar.close()

    # modify the feature value
    instances = feat_polished(instances)

    # convert the dict to tensor to learn the model
    feat_dicts = []
    for instance in instances:
        # ensure the order is the same as in the later part
        keys = sorted([key for key in instance.keys() if key.endswith("tweettext")])
        for key in keys:
            featkey = key.split("_")[0] + "_addfeat"
            feat_dicts.append(instance[featkey])
    dv = DictVectorizer(sparse=False)
    feat_vectorized = dv.fit_transform(feat_dicts)
    for index_outside, instance in enumerate(instances):
        small_feats = feat_vectorized[index_outside * 7:(index_outside + 1) * 7]
        # ensure the order is the same as the previous part
        keys = sorted([key for key in instance.keys() if key.endswith("tweettext")])
        for index_inside, key in enumerate(keys):
            newfeatkey = key.split("_")[0] + "_addfeattensor"
            feattensor = torch.FloatTensor(small_feats[index_inside]).unsqueeze(0).to('cpu')
            instance[newfeatkey] = feattensor

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
    :return: instances with bert output
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
        pbar = tqdm(total=len(instances))
        for index, instance in enumerate(instances):
            with torch.no_grad():
                small_padded_ids = all_padded_ids[index].unsqueeze(0).to(device)
                small_attention_mask = all_attention_mask[index].unsqueeze(0).to(device)
                last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

            # after getting the representation, move the tensor to CPU to free GPU memory
            anchor_bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
            instance['anchor_bertoutput'] = anchor_bert_feature
            pbar.update(1)
        pbar.close()

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
        pbar = tqdm(total=len(instances))
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
            pbar.update(1)
        pbar.close()

    # clear cache to free GPU memory
    torch.cuda.empty_cache()

    return instances


def add_vgg_output(instances, anchor_only):
    """
    add vgg output to the instances
    :param instances: instances without vgg output
    :param anchor_only: only extract feature of anchor tweet or anchor+context
    :return: instances with vgg output
    """

    # Use the GPU, if available, to get the image representation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the pretrained VGG16 (discard the last classification layer)
    vgg16_model = models.vgg16(pretrained=True)
    for p in vgg16_model.parameters():
        p.requires_grad = False
    vgg16_model = vgg16_model.to(device)
    vgg16_model.eval()

    pbar = tqdm(total=len(instances))
    for instance in instances:

        # get the path of image file
        filepath = instance['anchor_imagepath']

        # preprocess the image, convert it into RGB format if it is not RGB
        input_image = Image.open(filepath).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)

        # get the image representation
        with torch.no_grad():

            output = vgg16_model(input_tensor)
            # move the tensor to the CPU to free GPU memory
            output = output.to('cpu')

        # add image representation into the dictionary
        instance['anchor_vggoutput'] = output

        if not anchor_only:

            for i in range(8, 14):

                # get the path of image file if the image exists
                imagekey = f"context{i}_imagepath"
                if imagekey in instance.keys():
                    filepath = instance[imagekey]

                    # preprocess the image, convert it into RGB format if it is not RGB
                    input_image = Image.open(filepath).convert('RGB')
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

                    # get the image representation
                    with torch.no_grad():
                        output = vgg16_model(input_tensor)
                        # move the tensor to the CPU to free GPU memory
                        output = output.to('cpu')

                    # add image representation into the dictionary
                    instance[f"context{i}_vggoutput"] = output

        pbar.update(1)
    pbar.close()

    # clear cache to free GPU memory
    torch.cuda.empty_cache()

    return instances


def get_final_feats(instance):
    """
    get the input of the network based on mode (for simple_nn)
    :param instance: instance
    :return: input of the network
    """
    task_type = instance['tasktype']
    if task_type == 'anchor_text_only':
        feat = instance['anchor_bertoutput']
    if task_type == 'anchor_image_only':
        feat = instance['anchor_vggoutput']
    if task_type == 'anchor_text_image':
        feat = torch.cat((instance['anchor_bertoutput'], instance['anchor_vggoutput']), dim=1)
    if task_type in ['all_bert_only', 'all_bert_lstm', 'all_bert_lstm_onlybefore', 'all_bert_lstm_onlyafter', 'all_bert_lstm_noaddfeat']:
        feat = torch.cat((
            instance['context8_bertoutput'],
            instance['context9_bertoutput'],
            instance['context10_bertoutput'],
            instance['anchor_bertoutput'],
            instance['context11_bertoutput'],
            instance['context12_bertoutput'],
            instance['context13_bertoutput'],
        ), dim=1)
        # check if additional features exist
        if 'anchor_addfeattensor' in instance.keys():
            feat = torch.cat((feat, instance['context8_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['context9_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['context10_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['anchor_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['context11_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['context12_addfeattensor']), dim=1)
            feat = torch.cat((feat, instance['context13_addfeattensor']), dim=1)
    return feat


class TweetDataset(Dataset):

    # write a new class to load our dataset
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, index):
        instance = self.instances[index]
        label = instance['adjudicated_label']
        feat = get_final_feats(instance)
        return feat, label

    def __len__(self):
        return len(self.instances)


def get_data_loader(train_instances, dev_instances, test_instances, batch_size):
    """
    get the data loaders for train, development, and test
    :param train_instances: the training data
    :param dev_instances: the development data
    :param test_instances: the test data
    :param batch_size: batch_size
    :return: data loaders for train, development, and test
    """
    train_dataset = TweetDataset(train_instances)
    dev_dataset = TweetDataset(dev_instances)
    test_dataset = TweetDataset(test_instances)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, dev_loader, test_loader

