"""
Baseline Siamese 2D CNN, followed by an LSTM.
"""

import numpy as np
import torch
from torch import nn

from models.baseline import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetLSTM(SiamNet):
    def __init__(self, augmentation=None,
                 adam=False, momentum=0.9, weight_decay=0.0005, dropout_rate=0.5, include_cov=False,
                 lr=0.005, output_dim=256, weighted_loss=0.5,
                 n_lstm_layers=1, hidden_dim=256, bidirectional=False, insert_where=0, model_hyperparams=None):
        self.save_hyperparameters()
        if model_hyperparams is not None:
            super().__init__(augmentation=augmentation, model_hyperparams=model_hyperparams)
        else:
            super().__init__(augmentation=augmentation, model_hyperparams=self.hparams)

        # Change linear layers
        if self.hparams.insert_where == 0:
            # after first FC layer
            input_size = 1024
            self.fc9.fc9 = nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)
        elif self.hparams.insert_where == 1:
            # after conv layers, before first FC layer
            input_size = 256 * 3 * 3 * 2
            self.fc8.fc8 = nn.Linear(self.hparams.hidden_dim, 1024)
        else:
            # before last FC layer
            input_size = self.hparams.output_dim
            self.fc10.fc10 = nn.Linear(self.hparams.hidden_dim, 2)

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
            out = self._cnn_lstm_0(data)
        elif self.hparams.insert_where == 1:
            out = self._cnn_lstm_1(data)
        else:
            out = self._cnn_lstm_2(data)

        return torch.log_softmax(out, dim=1)

    def _cnn_lstm_0(self, data):
        """Default forward pass.
        Assumes no covariates. Follows CNN -> first FC layer -> concat views ->  LSTM -> remaining FC
        """
        x_t = data['img']
        x = torch.div(x_t, 255)

        if len(x.size()) == 5:
            x = x_t[0]

        T, C, H, W = x.size()
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
            z = z.view([T, 1, -1])
            z = self.fc8(z)
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(1, T, -1)

        lstm_out, (h_f, _) = self.lstm(x)
        x = h_f[0]

        x = self.fc9(x)
        x = self.fc10(x)
        return x

    def _cnn_lstm_1(self, data):
        """Alternative forward pass.
        Assumes no covariates. Follows CNN -> concat views ->  LSTM -> remaining FC.
        """
        x_t = data['img']
        x = torch.div(x_t, 255)

        if len(x.size()) == 5:
            x = x_t[0]

        T, C, H, W = x.size()
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
            z = z.view([T, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(1, T, -1)

        lstm_out, (h_f, _) = self.lstm(x)
        x = h_f[0]

        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        return x

    def _cnn_lstm_2(self, data):
        """Alternative forward pass.
        Assumes no covariates. Follows CNN -> FC -> LSTM -> last (prediction) FC."""
        x_t = data['img']
        x = torch.div(x_t, 255)

        if len(x.size()) == 5:
            x = x_t[0]

        T, C, H, W = x.size()
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
            z = z.view([T, 1, -1])
            z = self.fc8(z)
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view((T, 1, -1))

        x = self.fc9(x)

        x = x.view(1, T, -1)
        lstm_out, (h_f, _) = self.lstm(x)
        x = h_f[0]

        x = self.fc10(x)
        return x


if __name__ == '__main__':
    import numpy as np

    hyperparams = {'lr': 0.001, "batch_size": 1,
                   'adam': True,
                   'momentum': 0.9,
                   'weight_decay': 0.0005,
                   'include_cov': False,
                   'output_dim': 128,
                   'dropout_rate': 0,
                   'weighted_loss': 0.5,
                   'stop_epoch': 40
                   }
    model = SiamNetLSTM(hyperparams, insert_where=2)


    def simulate(n=5):
        data = {"img": torch.from_numpy(np.random.rand(1, n, 2, 256, 256)).type(torch.FloatTensor)}
        labels = np.random.randint(0, 2, n)
        # output_ = model.forward(data)
        # reducer_ = umap.UMAP(random_state=42)
        # embeds_ = reducer_.fit_transform(output_)
        # plot_umap(embeds_, labels)
        print(model.forward(data))


    simulate()
