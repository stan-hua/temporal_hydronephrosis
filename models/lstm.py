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
                 batch_size=1,
                 n_lstm_layers=1, hidden_dim=256,
                 bidirectional=False, insert_where=2):
        super().__init__(classes=classes, num_inputs=num_inputs, output_dim=output_dim, cov_layers=cov_layers,
                         device=device, dropout_rate=dropout_rate)

        # LSTM parameters
        self.batch_size = batch_size
        self.insert_where = insert_where
        self.n_lstm_layers = n_lstm_layers

        self.hidden_dim = hidden_dim

        # Change FC layers
        if self.insert_where == 0:       # immediately after U-Net
            input_size = 256 * 7 * 7 * 2
            self.fc6c.fc7 = nn.Linear(self.hidden_dim, 512)
        elif self.insert_where == 1:      # after first FC layer
            input_size = 1024
            self.fc7_new.fc7 = nn.Linear(self.hidden_dim, self.output_dim)
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

        @param x_t: tuple containing x_t, and length of each sequence in x_t
        @param x_lens:
        """
        x_t, x_lengths = data
        return self.embed_after_fc7_new(x_t, x_lengths)

    def embed_after_fc7_new(self, x_t, x_lengths):
        """Alternative forward pass. LSTM placed after fc7_new layer"""
        t_embeddings = []

        for x in x_t:
            if self.cov_layers:
                in_dict = x
                x = in_dict['img']

            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)

                # Grayscale to RGB
                curr_x = curr_x.expand(-1, 1, -1, -1)
                if torch.cuda.is_available():
                    input_ = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input_ = torch.FloatTensor(curr_x.to(self.device))
                out1 = self.conv1(input_)
                out2 = self.conv2(out1)
                out3 = self.conv3(out2)
                out4 = self.conv4(out3)
                out5 = self.conv5(out4)
                out6 = self.fc6(out5)
                unet1 = self.uconnect1(out6)

                curr_x = self.fc6c(unet1.view([B, 1, -1]))
                x_list.append(curr_x.view([B, 1, -1]))

            x = torch.cat(x_list, 1)
            # x = torch.sum(x, 1)
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
        return pred

    # TODO: Do this
    def embed_before_fc6(self, x_t):
        """Alternative forward pass. LSTM placed before fc6 layer"""
        t_embeddings = []

        for x in x_t:
            if self.cov_layers:
                in_dict = x
                x = in_dict['img']

            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)

                # Grayscale to RGB
                curr_x = curr_x.expand(-1, 1, -1, -1)
                if torch.cuda.is_available():
                    input_ = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input_ = torch.FloatTensor(curr_x.to(self.device))
                out1 = self.conv1(input_)
                out2 = self.conv2(out1)
                out3 = self.conv3(out2)
                out4 = self.conv4(out3)
                out5 = self.conv5(out4)
                out6 = self.fc6(out5)
                unet1 = self.uconnect1(out6)

                x_list.append(unet1.view([B, 1, -1]))

            x = torch.cat(x_list, 1)
            t_embeddings.append(x)

        x = torch.stack(t_embeddings).unsqueeze(0)

        # initiate batch hidden and cell states
        hidden_0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).requires_grad_().to(device)
        cellState_0 = torch.zeros(self.n_lstm_layers, self.batch_size, self.hidden_dim).requires_grad_().to(device)

        lstm_out, _ = self.lstm(x, (hidden_0.detach(), cellState_0.detach()))
        lstm_out = torch.squeeze(lstm_out, 0)

        x = self.fc6c(lstm_out.view([B, 1, -1]))
        x = self.fc7_new(x)

        pred = self.classifier_new(x)
        return pred

    # TODO: Do this
    def embed_after_fc6(self, x_t):
        raise NotImplementedError
