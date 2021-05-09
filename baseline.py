import argparse
import os
from utils import preprocess
from sklearn.metrics import classification_report


def main(mode):

    data_dir = 'data'
    annotation_filename = "new_annot.json"
    annotation_dir = os.path.join(data_dir, 'annotations', annotation_filename)

    # load data
    instances = preprocess.load_data(annotation_dir, mode)

    # add majority baseline output
    instances = preprocess.add_baseline_output(instances)

    # split the instances into train, dev, and test
    train_instances, dev_instances, test_instances = preprocess.split_instances(instances)

    # make predication and evaluate
    gold_labels = [x['adjudicated_label'] for x in test_instances]
    pred_labels = [x['baseline_output'] for x in test_instances]
    print(classification_report(gold_labels, pred_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', help='anchor_text_only or anchor_image_only or anchor_text_image')
    args = parser.parse_args()

    main(mode=args.mode)
