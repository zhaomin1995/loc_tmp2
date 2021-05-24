import torch
import torch.nn as nn
import torch.nn.functional as F

# parameter setting
bert_feat_dim = 768
lstm_dim = 1024
output_dim = 2
hidden_dim = 512
dropout_rate = 0.5


class ComplicatedBertLSTM(nn.Module):

    def __init__(self, additional_feat_dim=0):
        super(ComplicatedBertLSTM, self).__init__()
        self.bert_feat_dim = bert_feat_dim
        self.additional_feat_dim = additional_feat_dim
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert_feat_dim, self.lstm_dim, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(self.lstm_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_combined):

        # split the combined feature
        context1_feat = feat_combined[:, :self.bert_feat_dim * 1].unsqueeze(1)
        context2_feat = feat_combined[:, self.bert_feat_dim * 1:self.bert_feat_dim * 2].unsqueeze(1)
        context3_feat = feat_combined[:, self.bert_feat_dim * 2:self.bert_feat_dim * 3].unsqueeze(1)
        anchor_feat = feat_combined[:, self.bert_feat_dim * 3:self.bert_feat_dim * 4].unsqueeze(1)
        context4_feat = feat_combined[:, self.bert_feat_dim * 4:self.bert_feat_dim * 5].unsqueeze(1)
        context5_feat = feat_combined[:, self.bert_feat_dim * 5:self.bert_feat_dim * 6].unsqueeze(1)
        context6_feat = feat_combined[:, self.bert_feat_dim * 6:self.bert_feat_dim * 7].unsqueeze(1)

        # prepare for the input of LSTM
        lstm_input = torch.cat((
            context1_feat,
            context2_feat,
            context3_feat,
            anchor_feat,
            context4_feat,
            context5_feat,
            context6_feat
        ), dim=1)

        # pass the LSTM
        out, _ = self.lstm(lstm_input)

        # only take the last hidden state
        out = out[:, -1, :]

        # pass the fully-connected layer(s)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
