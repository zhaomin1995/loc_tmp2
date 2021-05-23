import torch.nn as nn
import torch.nn.functional as F


# parameter setting
bert_feat_dim = 768
output_dim = 2
hidden_dim = 1024
dropout_rate = 0.5


class ComplicatedBertOnly(nn.Module):

    def __init__(self, additional_feat_dim=0):
        super().__init__()
        self.bert_feat_dim = bert_feat_dim
        self.additional_feat_dim = additional_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(self.bert_feat_dim*7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_combined):

        # split the combined feature
        bert_feat = feat_combined[:, :self.bert_feat_dim * 7]
        additional_feat = feat_combined[:, self.bert_feat_dim * 7:]

        # pass fully-connected layer
        out = self.fc1(bert_feat)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
