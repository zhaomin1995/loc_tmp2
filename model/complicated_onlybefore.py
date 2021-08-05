import torch
import torch.nn as nn
import torch.nn.functional as F

# parameter setting
bert_feat_dim = 768
lstm_dim = 1024
output_dim = 2
hidden_dim = 512
dropout_rate = 0.5


class ComplicatedOnlyBefore(nn.Module):

    def __init__(self, additional_feat_dim=0):
        super(ComplicatedOnlyBefore, self).__init__()
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
        context1_feat = feat_combined[:, :self.bert_feat_dim * 1].unsqueeze(1)
        context2_feat = feat_combined[:, self.bert_feat_dim * 1:self.bert_feat_dim * 2].unsqueeze(1)
        context3_feat = feat_combined[:, self.bert_feat_dim * 2:self.bert_feat_dim * 3].unsqueeze(1)
        anchor_feat = feat_combined[:, self.bert_feat_dim * 3:self.bert_feat_dim * 4].unsqueeze(1)

        # get the additional features
        addfeat_dim = int((feat_combined.shape[1] - self.bert_feat_dim * 7) / 7)
        context1_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 0:self.bert_feat_dim * 7 + addfeat_dim * 1].unsqueeze(1)
        context2_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 1:self.bert_feat_dim * 7 + addfeat_dim * 2].unsqueeze(1)
        context3_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 2:self.bert_feat_dim * 7 + addfeat_dim * 3].unsqueeze(1)
        anchor_addfeat = feat_combined[:, self.bert_feat_dim * 7 + addfeat_dim * 3:self.bert_feat_dim * 7 + addfeat_dim * 4].unsqueeze(1)

        # prepare for the input of LSTM
        lstm_input = torch.cat((
            torch.cat((context1_feat, context1_addfeat), dim=2),
            torch.cat((context2_feat, context2_addfeat), dim=2),
            torch.cat((context3_feat, context3_addfeat), dim=2),
            torch.cat((anchor_feat, anchor_addfeat), dim=2),
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
