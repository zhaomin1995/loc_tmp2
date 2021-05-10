import torch
import torch.nn as nn
import torch.nn.functional as F

# parameter setting
bert_feat_dim = 768
vgg_feat_dim = 1000
output_dim = 2
hidden_dim = 512
dropout_rate = 0.2


class AnchorImageOnlyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert_feat_dim = bert_feat_dim
        self.vgg_feat_dim = vgg_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.fc_bert = nn.Linear(bert_feat_dim, hidden_dim)
        self.fc_vgg = nn.Linear(vgg_feat_dim, hidden_dim)
        self.fc_both = nn.Linear(bert_feat_dim+vgg_feat_dim, hidden_dim)
        self.fc_last1 = nn.Linear(hidden_dim*3, hidden_dim)
        self.fc_last2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_combined):
        bert_feat = feat_combined[:, :self.bert_feat_dim]
        vgg_feat = feat_combined[:, self.bert_feat_dim:self.bert_feat_dim+self.vgg_feat_dim]
        both_feat = torch.cat((bert_feat, vgg_feat), dim=1)
        bert_out = self.fc_bert(bert_feat)
        vgg_out = self.fc_vgg(vgg_feat)
        both_out = self.fc_both(both_feat)
        combined_out = torch.cat((bert_out, vgg_out, both_out), dim=1)
        out = self.fc_last1(combined_out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_last2(out)
        return out
