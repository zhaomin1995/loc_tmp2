import torch.nn as nn
import torch.nn.functional as F

# parameter setting
vgg_feat_dim = 1000
output_dim = 2
hidden_dim = 512
dropout_rate = 0.5


class AnchorImageOnlyModel(nn.Module):

    def __init__(self):
        super(AnchorImageOnlyModel, self).__init__()
        self.vgg_feat_dim = vgg_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(self.vgg_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vgg_feat):
        out = self.fc1(vgg_feat)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
