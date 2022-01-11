"""Baseline Siamese 2D CNN model with Conv Pooling.
"""

import numpy as np
import torch
from torch import nn

from models.baseline_pl import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetConvPooling(SiamNet):
    def forward(self, data):
        """Accepts sequence of dual view images. Extracts penultimate layer embeddings for each dual view, then performs
        max pooling over time.
        """
        x_t = data['img']
        x_t = x_t.transpose(0, 1)

        t_embeddings = []
        for t, x in enumerate(x_t):
            B, T, C, H = x.size()
            x = x.transpose(0, 1)

            x_list = []
            for i in range(self.num_inputs):
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

            if self.cov_layers:
                x = self.fc9(x)
                x = self.fc10(x)

                age = data['Age_wks'].view(B, 1)
                side = data['Side_L'].view(B, 1)

                x = torch.cat((x, age, side), 1)
                x = self.fc10b(x)

            t_embeddings.append(x)

        x, _ = torch.max(torch.stack(t_embeddings), dim=0)

        if not self.cov_layers:
            x = self.fc9(x)
            x = self.fc10(x)
        else:
            x = self.fc10c(x)

        return torch.log_softmax(x, dim=1)

    @torch.no_grad()
    def forward_embed(self, data):
        x_t = data['img']
        x_t = x_t.transpose(0, 1)

        t_embeddings = []
        for t, x in enumerate(x_t):
            B, T, C, H = x.size()
            x = x.transpose(0, 1)

            x_list = []
            for i in range(self.num_inputs):
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

            if self.cov_layers:
                x = self.fc9(x)
                x = self.fc10(x)

                age = data['Age_wks'].view(B, 1)
                side = data['Side_L'].view(B, 1)

                x = torch.cat((x, age, side), 1)
                x = self.fc10b(x)

            t_embeddings.append(x)

        x, _ = torch.max(torch.stack(t_embeddings), dim=0)
        return x.cpu().detach().numpy()
