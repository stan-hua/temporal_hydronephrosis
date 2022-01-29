"""
Baseline Siamese 2D CNN, followed by an LSTM.
"""

import numpy as np
import torch
from torch import nn

from models.baseline_pl import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiameseLSTM(SiamNet):
    def __init__(self, model_hyperparams, n_lstm_layers=1, hidden_dim=256, bidirectional=False, insert_where=None):
        super().__init__(model_hyperparams)
        self.save_hyperparameters("n_lstm_layers", "hidden_dim", "bidirectional", "insert_where")

        # Change linear layers
        if self.hparams.insert_where == 0:
            # immediately after convolutional layer
            input_size = 256 * 3 * 3 * 2
            self.fc8.fc8 = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)
        elif self.hparams.insert_where == 1:
            # after first FC layer
            input_size = 1024
            self.fc7_new.fc7 = nn.Linear(256 * 3 * 3 * 2, self.hparams.output_dim)
        else:
            # right before prediction layer
            input_size = self.hparams.output_dim
            i = (1 if not self.hparams.bidirectional else 2)
            if not self.hparams.include_cov:
                self.fc9.fc9 = nn.Linear(self.hparams.hidden_dim * i, self.hparams.output_dim)
            else:
                self.fc10c.fc10c = nn.Linear(self.hparams.hidden_dim * i, 2)

        # LSTM layers
        self.lstm = nn.Sequential()
        self.lstm.add_module(f"lstm{1}", nn.LSTM(input_size, self.hparams.hidden_dim,
                                                 batch_first=True,
                                                 num_layers=self.hparams.n_lstm_layers,
                                                 bidirectional=self.hparams.bidirectional))

    def forward(self, data):
        """Accepts sequence of dual view images. Extracts penultimate layer embeddings for each dual view, then
        uses an LSTM to aggregate spatial features over time.
        """
        if self.hparams.insert_where == 0:
            out = self._cnn_lstm_2(data)
        else:
            out = self._cnn_lstm_0(data)

        return torch.log_softmax(out, dim=1)

    def _cnn_lstm_0(self, data):
        """Default forward pass.
        If no covariates, CNN -> first FC layer -> concat views ->  LSTM -> remaining FC.
        If there are covariates, CNN -> FC -> concat views -> FC (cov) -> LSTM -> remaining FC.
        """
        x_t = data['img']
        x_t = x_t.transpose(0, 1)

        t_embeddings = []
        for t, x in enumerate(x_t):
            B, V, H, W = x.size()
            x = x.transpose(0, 1)

            x_list = []
            for i in range(2):
                z = torch.unsqueeze(x[i], 1)
                z = z.expand(-1, 3, -1, -1)
                z = self.conv1(z)
                z = self.conv2(z)
                z = self.conv3(z)
                z = self.conv4(z)
                z = self.conv5(z)
                z = self.conv6(z)
                z = self.conv7(z)
                z = z.view([B, 1, -1])
                z = self.fc8(z)
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = x.view(B, -1)

            if self.hparams.include_cov:
                x = self.fc9(x)
                x = self.fc10(x)

                age = data['Age_wks'][:, t].view(B, 1)
                side = data['Side_L'][:, t].view(B, 1)

                x = torch.cat((x, age, side), 1)
                x = self.fc10b(x)

            t_embeddings.append(x)

        x = torch.stack(t_embeddings)

        print("Stack:", x.size())

        lstm_out, (h_f, _) = self.lstm(x)

        print("LSTM Out:", lstm_out.size())
        print("LSTM Out:", lstm_out[-1].size())
        print("Last Hidden State:", h_f.size())
        x = lstm_out[-1]                         # extract last hidden state

        if not self.hparams.include_cov:
            x = self.fc9(x)
            x = self.fc10(x)
        else:
            x = self.fc10c(x)

        print(x)
        return x

    # TODO: Do this
    def _cnn_lstm_1(self, x_t):
        """Alternative forward pass.
        If no covariates, CNN -> concat views ->  LSTM -> remaining FC.
        If there are covariates, CNN -> FC -> concat views -> FC (cov) -> LSTM -> remaining FC.
        """
        x_t = data['img']
        x_t = x_t.transpose(0, 1)

        t_embeddings = []
        for t, x in enumerate(x_t):
            B, V, H, W = x.size()
            x = x.transpose(0, 1)

            x_list = []
            for i in range(2):
                z = torch.unsqueeze(x[i], 1)
                z = z.expand(-1, 3, -1, -1)
                z = self.conv1(z)
                z = self.conv2(z)
                z = self.conv3(z)
                z = self.conv4(z)
                z = self.conv5(z)
                z = self.conv6(z)
                z = self.conv7(z)
                z = z.view([B, 1, -1])
                z = self.fc8(z)
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = x.view(B, -1)

            if self.hparams.include_cov:
                x = self.fc9(x)
                x = self.fc10(x)

                age = data['Age_wks'].view(B, 1)
                side = data['Side_L'].view(B, 1)

                x = torch.cat((x, age, side), 1)
                x = self.fc10b(x)

            t_embeddings.append(x)

        x = torch.stack(t_embeddings)
        lstm_out, (h_f, _) = self.lstm(x)
        x = h_f[-1]                         # extract last hidden state

        if not self.hparams.include_cov:
            x = self.fc9(x)
            x = self.fc10(x)
        else:
            x = self.fc10c(x)

        return x

    # TODO: Do this
    def _cnn_lstm_2(self, x_t):
        """Alternative forward pass. LSTM placed after last convolutional layer (fc6b)."""
        pass