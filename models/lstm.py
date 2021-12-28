"""Baseline Siamese 2D CNN, followed by an LSTM.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.baselineSiamese import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiameseLSTM(SiamNet):
    def __init__(self, classes=2, num_inputs=2, output_dim=128, cov_layers=False, device=None, dropout_rate=0.5,
                 batch_size=1, n_lstm_layers=1, hidden_dim=256, bidirectional=False, insert_where=2):
        super().__init__(classes=classes, num_inputs=num_inputs, output_dim=output_dim, cov_layers=cov_layers,
                         device=device, dropout_rate=dropout_rate)

        # LSTM parameters
        self.batch_size = batch_size
        self.insert_where = insert_where
        self.n_lstm_layers = n_lstm_layers

        self.hidden_dim = hidden_dim

        # Change FC layers
        if self.insert_where == 0:              # immediately after convolutional layer
            input_size = 256 * 3 * 3 * 2
        elif self.insert_where == 1:            # after first FC layer
            input_size = 1024
            self.fc7_new.fc7 = nn.Linear(256 * 3 * 3 * 2, self.output_dim)
        else:                                   # right before prediction layer
            input_size = self.output_dim
            self.classifier_new.fc8 = nn.Linear(self.hidden_dim, classes)

        # LSTM layers
        self.lstm = nn.Sequential()
        # for i in range(1, n_lstm_layers + 1):
        #     curr_input_size = input_size if i == 0 else hidden_size
        #     curr_hidden_size = hidden_size if i != (n_lstm_layers - 1) else output_hidden_dim
        #
        #     self.lstm.add_module(f"lstm{i}", nn.LSTM(curr_input_size, curr_hidden_size, 1,
        #                                              batch_first=True, bidirectional=bidirectional))

        self.lstm.add_module(f"lstm{1}", nn.LSTM(input_size, hidden_dim,
                                                 batch_first=True,
                                                 num_layers=n_lstm_layers,
                                                 bidirectional=bidirectional))

    def forward(self, data):
        """Accepts sequence of dual view images. Extracts penultimate layer embeddings for each dual view, then
        uses an LSTM to aggregate spatial features over time.

        @param data: tuple containing sequence of images X, their lengths, and optionally covariates.
        """
        if self.cov_layers:
            data, in_dict = data
            x_t, x_lengths = data
            x_t = x_t, in_dict
        else:
            x_t, x_lengths = data

        if self.insert_where == 0:
            out = self._embed_after_conv(x_t, x_lengths)
        else:
            out = self._embed_before_prediction(x_t, x_lengths)

        return out

    def _embed_before_prediction(self, x_t, x_lengths):
        """Alternative forward pass. LSTM placed after fc7_new layer."""
        if self.cov_layers:
            x_t, in_dict = x_t

        t_embeddings = []
        for x in x_t:
            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)

                # Grayscale to RGB
                curr_x = curr_x.expand(-1, 3, -1, -1)
                if torch.cuda.is_available():
                    input_ = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input_ = torch.FloatTensor(curr_x.to(self.device))
                z = self.conv(input_)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])
                z = self.fc6c(z)
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = self.fc7_new(x.view(B, -1))
            t_embeddings.append(x)

        # Stack image embeddings for each item in sequence. Then pack to remove padding
        x = torch.stack(t_embeddings)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

        # LSTM
        lstm_out, (h_f, _) = self.lstm(x)

        # Unpack sequence
        # lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # print("Last padded LSTM output", lstm_out[:, -1, :].view([x_t.size()[0], -1]).size())

        # Use last hidden state
        pred = self.classifier_new(h_f[-1])

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

        return pred

    # TODO: Do this
    def _embed_after_first_fc(self, x_t):
        """Alternative forward pass. LSTM placed after the first fully connected layer"""
        if self.cov_layers:
            x_t, in_dict = x_t

        t_embeddings = []
        for x in x_t:
            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)
                curr_x = curr_x.expand(-1, 3, -1, -1)
                if torch.cuda.is_available():
                    input_ = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input_ = torch.FloatTensor(curr_x.to(self.device))
                z = self.conv(input_)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])
                # z = self.fc6c(z)
                # z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = self.fc7_new(x.view(B, -1))
            t_embeddings.append(x)

        x = torch.stack(t_embeddings)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (h_f, _) = self.lstm(x)
        pred = self.classifier_new(h_f[-1])

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

        return pred

    # TODO: Do this
    def _embed_after_conv(self, x_t, x_lengths):
        """Alternative forward pass. LSTM placed after last convolutional layer (fc6b)."""
        if self.cov_layers:
            x_t, in_dict = x_t

        t_embeddings = []
        for x in x_t:
            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)
                curr_x = curr_x.expand(-1, 3, -1, -1)
                if torch.cuda.is_available():
                    input_ = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input_ = torch.FloatTensor(curr_x.to(self.device))
                z = self.conv(input_)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])
                # z = self.fc6c(z)
                # z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = x.view(B, -1)
            # x = self.fc7_new(x)
            t_embeddings.append(x)

        x = torch.stack(t_embeddings)
        x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, (h_f, _) = self.lstm(x)
        pred = self.classifier_new(h_f[-1])

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

        return pred
