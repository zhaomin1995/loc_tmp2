import argparse
import time
import torch
from utils import preprocess
from utils import learning_helper


def main(mode):

    # load data
    data_dir = 'data/annotations/new_annot.json'
    instances = preprocess.load_data(data_dir, mode)

    # add BERT output and/or VGG output
    if mode == 'anchor_text_only' or mode == 'anchor_text_image':
        start_time = time.time()
        instances = preprocess.add_bert_output(instances, anchor_only=True)
        end_time = time.time()
        elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)
        print(f"Time spent for BERT: {elapsed_mins}m {elapsed_secs}s")
    if mode == 'anchor_image_only' or mode == 'anchor_text_image':
        start_time = time.time()
        instances = preprocess.add_vgg_output(instances, anchor_only=True)
        end_time = time.time()
        elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)
        print(f"Time spent for VGG: {elapsed_mins}m {elapsed_secs}s")

    # split instances into train, dev, and test
    train_instances, dev_instances, test_instances = preprocess.split_instances(instances)

    # get the data loader of train, dev, and test
    train_loader, dev_loader, test_loader = preprocess.get_data_loader(train_instances, dev_instances, test_instances)

    # check if we can use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the model based on mode
    classifier = learning_helper.get_model(mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', help='anchor_text_only or anchor_image_only or anchor_text_image')
    args = parser.parse_args()

    main(mode=args.mode)
