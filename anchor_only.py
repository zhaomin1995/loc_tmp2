import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import preprocess
from utils import learning_helper
from utils import evaluator
from sklearn.metrics import classification_report


# define some global parameters
num_epochs = 100
batch_size = 16
patience = 5
learning_rate = 1e-02


def main(mode, model_type):

    # load data
    data_dir = 'data/annotations/new_annot.json'
    instances = preprocess.load_data(data_dir, mode)

    # add BERT output and/or VGG output
    if mode == 'anchor_text_only' or mode == 'anchor_text_image':
        print("Extracting textual features using BERT ...")
        start_time = time.time()
        instances = preprocess.add_bert_output(instances, anchor_only=True)
        end_time = time.time()
        elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)
        print(f"Time spent for BERT: {elapsed_mins}m {elapsed_secs}s")
    if mode == 'anchor_image_only' or mode == 'anchor_text_image':
        print("Extracting image features using VGG ...")
        start_time = time.time()
        instances = preprocess.add_vgg_output(instances, anchor_only=True)
        end_time = time.time()
        elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)
        print(f"Time spent for VGG: {elapsed_mins}m {elapsed_secs}s")

    # split instances into train, dev, and test
    train_instances, dev_instances, test_instances = preprocess.split_instances(instances)

    # get the data loader of train, dev, and test
    train_loader, dev_loader, test_loader = preprocess.get_data_loader(train_instances, dev_instances, test_instances, batch_size)

    # define the label mapping
    label_to_idx = {'Yes': 1, 'No': 0}
    idx_to_label = {1: 'Yes', 0: 'No'}

    # check if we can use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'retrain':

        # get the model based on mode and move model to GPU is GPU is available
        classifier = learning_helper.get_model(mode)
        classifier = classifier.to(device)

        # define the optimizer, loos function, and some parameters
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)

        # train the model
        best_valid_loss = float('inf')
        check_stopping = 0
        model_name = f'retrained_{mode}_classifier.pkl'
        model_path = os.path.join('retrained_models', model_name)
        for i in range(num_epochs):

            start_time = time.time()
            train_loss, train_acc = learning_helper.train(classifier, train_loader, optimizer, criterion, device, label_to_idx)
            dev_loss, dev_acc = learning_helper.evaluate(classifier, dev_loader, criterion, device, label_to_idx)
            end_time = time.time()
            elapsed_mins, elapsed_secs = learning_helper.epoch_time(start_time, end_time)

            print("-" * 60)
            print(f"Epoch: {i + 1} || Epoch Time: {elapsed_mins}m {elapsed_secs}s")
            print(f"Epoch: {i + 1} || Train loss: {train_loss:.02f}, Train Acc: {train_acc:.02f}")
            print(f"Epoch: {i + 1} || Dev loss: {dev_loss:.02f}, Dev Acc: {dev_acc:.02f}")

            # check if we need to save the model
            if dev_loss < best_valid_loss:
                check_stopping = 0
                best_valid_loss = dev_loss
                torch.save(classifier, model_path)
            else:
                check_stopping += 1
                print(f"The loss on development set does not decrease")
                if check_stopping == patience:
                    print("The loss on development set does not decrease, stop training!")
                    break

    if model_type == 'pretrained':

        model_name = f'pretrained_{mode}_classifier.pkl'
        model_path = os.path.join('pretrained_models', model_name)
        classifier = torch.load(model_path)
        classifier = classifier.to(device)

    classifier.eval()
    pred_labels = evaluator.test_model(classifier, test_loader, idx_to_label, device)
    gold_labels = [x['adjudicated_label'] for x in test_instances]
    print('-' * 60)
    print(classification_report(gold_labels, pred_labels))

    # save the predicted label
    evaluator.save_prediction(mode, pred_labels, test_instances)
    print("The prediction is saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', help='anchor_text_only or anchor_image_only or anchor_text_image')
    parser.add_argument('-model_type', '--model_type', help='retrain or pretrained')
    args = parser.parse_args()

    main(mode=args.mode, model_type=args.model_type)
