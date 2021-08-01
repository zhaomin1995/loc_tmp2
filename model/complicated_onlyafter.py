import torch
import torch.nn as nn
import torch.nn.functional as F

# parameter setting
bert_feat_dim = 768
lstm_dim = 1024
output_dim = 2
hidden_dim = 512
dropout_rate = 0.5


class ComplicatedOnlyAfter(nn.Module):

    def __init__(self, additional_feat_dim=0):
        super(ComplicatedOnlyAfter, self).__init__()
        self.bert_feat_dim = bert_feat_dim
        self.additional_feat_dim = additional_feat_dim
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.bert_feat_dim + self.additional_feat_dim, self.lstm_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.lstm_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_combined):
        # get the bert output
        anchor_feat = feat_combined[:, self.bert_feat_dim * 3:self.bert_feat_dim * 4].unsqueeze(1)
        context4_feat = feat_combined[:, self.bert_feat_dim * 4:self.bert_feat_dim * 5].unsqueeze(1)
        context5_feat = feat_combined[:, self.bert_feat_dim * 5:self.bert_feat_dim * 6].unsqueeze(1)
        context6_feat = feat_combined[:, self.bert_feat_dim * 6:self.bert_feat_dim * 7].unsqueeze(1)

        # get the additional features
        addfeat_dim = int((feat_combined.shape[1] - self.bert_feat_dim * 7) / 7)
        anchor_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 3:self.bert_feat_dim * 7 + addfeat_dim * 4].unsqueeze(1)
        context4_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 4:self.bert_feat_dim * 7 + addfeat_dim * 5].unsqueeze(1)
        context5_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 5:self.bert_feat_dim * 7 + addfeat_dim * 6].unsqueeze(1)
        context6_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 6:self.bert_feat_dim * 7 + addfeat_dim * 7].unsqueeze(1)

        # prepare for the input of LSTM
        lstm_input = torch.cat((
            torch.cat((anchor_feat, anchor_addfeat), dim=2),
            torch.cat((context4_feat, context4_addfeat), dim=2),
            torch.cat((context5_feat, context5_addfeat), dim=2),
            torch.cat((context6_feat, context6_addfeat), dim=2),
        ), dim=1)

        # pass the LSTM
        lstm_output, _ = self.lstm(lstm_input)

        # global average pooling on the BiLSTM output
        lstm_output = nn.AvgPool1d(4, 4)(lstm_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)

        # pass the fully-connected layer(s)
        out = self.fc1(lstm_output)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
