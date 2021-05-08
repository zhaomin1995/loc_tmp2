import argparse
import os
from utils import preprocess
from sklearn.metrics import classification_report


def main(data_dir):
    annotation_filename = "new_annot.json"
    annotation_dir = os.path.join(data_dir, 'annotations', annotation_filename)
    instances = preprocess.load_data(annotation_dir)
    instances = preprocess.add_baseline_output(instances)
    gold_labels = [x['adjudicated_label'] for x in instances]
    pred_labels = [x['baseline_output'] for x in instances]
    print(classification_report(gold_labels, pred_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--dir', help='The path of the data folder')
    args = parser.parse_args()

    main(data_dir=args.dir)
