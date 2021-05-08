import json
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


