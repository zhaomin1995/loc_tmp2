import torch
import json
import os


def test_model(model, test_loader, idx_to_label, device):
    """
    get the prediction using the trained model and test data
    :param model: trained model
    :param test_loader: the data loader of test data
    :param idx_to_label: convert the index to the label
    :return: the predicted labels
    """

    # set the model as evaluation mode
    model.eval()

    pred_labels = []
    for test_x, test_y in test_loader:
        with torch.no_grad():

            # get the text representation
            feats = test_x.squeeze(1).to(device)

            preds = model(feats)
            pred_ids = preds.argmax(dim=1)
            for pred_id in pred_ids:
                pred_label = idx_to_label[pred_id.item()]
                pred_labels.append(pred_label)

    return pred_labels


def save_prediction(mode, pred_labels, test_instances, prediction_dir='predictions'):
    """
    save the predicted labels to do the error analysis
    :param mode: text_only OR image_only OR text_image
    :param pred_labels: predicted labels returned by train model
    :param test_instances: test instances
    :param prediction_dir:
    :return:
    """
    # save the predicted label
    prediction_filename = f'prediction_{mode}.json'
    prediction_filepath = os.path.join(prediction_dir, prediction_filename)
    with open(prediction_filepath, 'w+') as pred_file:
        for index, instance in enumerate(test_instances):
            # remove the bert_output and resnet_output

            for key in list(instance.keys()):
                if 'vggoutput' in key or 'bertoutput' in key:
                    del instance[key]

            # add the predicted label
            pred_label = pred_labels[index]
            instance['pred_label'] = pred_label

            # save the instance
            pred_file.write(json.dumps(instance))
            pred_file.write('\n')


