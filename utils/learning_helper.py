import torch
from model.anchor_text_only import AnchorTextOnlyModel
from model.anchor_image_only import AnchorImageOnlyModel
from model.anchor_text_image import AnchorTextImageModel
from model.complicated_bert_lstm_combined import ComplicatedBertLSTM
from model.complicated_onlybefore import ComplicatedOnlyBefore
from model.complicated_onlyafter import ComplicatedOnlyAfter


def epoch_time(start_time, end_time):
    """
    Compute the time spent for each epoch
    :param start_time: the time before each epoch
    :param end_time: the time after each epoch
    :return: the time spent for each epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_model(mode, additional_feat_dim=0):
    """
    get the initialized model based on mode
    :param mode: anchor_text_only or anchor_image_only or anchor_text_image
    :param additional_feat_dim: the dimensionality of additional features
    :return: initialized model (parameters are randomly initialized)
    """
    if mode == 'anchor_text_only':
        model = AnchorTextOnlyModel()
    if mode == 'anchor_image_only':
        model = AnchorImageOnlyModel()
    if mode == 'anchor_text_image':
        model = AnchorTextImageModel()
    if mode == 'all_bert_lstm' or mode == 'all_bert_lstm_noaddfeat':
        model = ComplicatedBertLSTM(additional_feat_dim=additional_feat_dim)
    if mode == 'all_bert_lstm_onlybefore':
        model = ComplicatedOnlyBefore(additional_feat_dim=additional_feat_dim)
    if mode == 'all_bert_lstm_onlyafter':
        model = ComplicatedOnlyAfter(additional_feat_dim=additional_feat_dim)
    return model


def train(classifier, iterator, optimizer, criterion, device, label_to_idx):
    """
    train the model
    :param classifier: the model to be trained
    :param iterator: training data loader
    :param optimizer: optimizer
    :param criterion: loss function
    :param device: cpu or cuda
    :param label_to_idx: convert the label into index to use the CrossEntropy
    :return: the loss and accuracy of each epoch
    """

    epoch_loss = 0
    epoch_correct = 0
    num_instances = 0

    # set the model as training mode
    classifier.train()

    for batch_x, batch_y in iterator:

        # get the text representation and image representation
        feats = batch_x.squeeze(1).to(device)

        label_ids = torch.tensor([label_to_idx[label] for label in batch_y]).to(device)

        optimizer.zero_grad()

        # use the model to predict
        preds = classifier(feats)
        preds = preds.squeeze(1).to(device)

        loss = criterion(preds, label_ids)
        epoch_loss += loss.item()

        # back-propogate
        loss.backward()
        optimizer.step()

        # calculate the accuracy
        pred_ids = preds.argmax(dim=1)
        epoch_correct += pred_ids.eq(label_ids).sum().item()
        num_instances += len(batch_y)

    return epoch_loss / num_instances, epoch_correct / num_instances


def evaluate(classifier, iterator, criterion, device, label_to_idx):
    """
    evaluate the trained model
    :param classifier: the trained model
    :param iterator: development data loader
    :param criterion: loss function
    :param device: cuda or cpu
    :param label_to_idx: convert the label into index to use the loss function
    :return: the loss and accuracy on development data
    """
    epoch_loss = 0
    epoch_correct = 0
    num_instances = 0

    # set the model as evaluation mode
    classifier.eval()

    with torch.no_grad():

        for batch_x, batch_y in iterator:

            # get the text representation and image representation
            feats = batch_x.squeeze(1).to(device)

            label_ids = torch.tensor([label_to_idx[label] for label in batch_y]).to(device)

            # use the model to predict
            preds = classifier(feats)
            preds = preds.squeeze(1).to(device)

            loss = criterion(preds, label_ids)
            epoch_loss += loss.item()

            # calculate the accuracy
            pred_ids = preds.argmax(dim=1)
            epoch_correct += pred_ids.eq(label_ids).sum().item()
            num_instances += len(batch_y)

    return epoch_loss / num_instances, epoch_correct / num_instances


