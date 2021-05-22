import torch.nn as nn
import torch.nn.functional as F

# parameter setting
bert_feat_dim = 768
output_dim = 2
hidden_dim = 512
dropout_rate = 0.5


class AnchorTextOnlyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert_feat_dim = bert_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(self.bert_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, bert_feat):
        out = self.fc1(bert_feat)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
