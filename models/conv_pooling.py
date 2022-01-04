"""Baseline Siamese 2D CNN model with Conv Pooling.
"""

import numpy as np
import torch
from torch import nn

from models.baseline import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetConvPooling(SiamNet):
    def forward(self, data):
        """Accepts sequence of dual view images. Extracts penultimate layer embeddings for each dual view, then performs
        max pooling over time.
        """
        if self.cov_layers:
            data, in_dict = data
            x_t, x_lengths = data
        else:
            x_t, x_lengths = data

        t_embeddings = []
        x_t = x_t.transpose(0, 1)
        for t, x in enumerate(x_t):
            if torch.mean(x) == 0:      # stop once hit zero padded image
                break

            if self.num_inputs == 1:
                x = x.unsqueeze(1)

            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                curr_x = torch.unsqueeze(x[i], 1)
                curr_x = curr_x.expand(-1, 3, -1, -1)
                z = self.conv(curr_x)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])

                z = self.fc6c(z)
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            x = x.view(B, -1)

            if self.cov_layers:
                age = torch.from_numpy(np.array([cov_dict_['Age_wks'][t] for cov_dict_ in in_dict])).type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)
                side = torch.from_numpy(np.array([cov_dict_['Side_L'][t] for cov_dict_ in in_dict])).type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)

                age = age.expand(-1, 512)  # (B, 512)
                side = side.expand(-1, 512)
                x = torch.cat((x, age, side), 1)

            t_embeddings.append(x)

        x, _ = torch.max(torch.stack(t_embeddings), dim=0)
        x = self.fc7_new(x)
        x = self.fc8(x)
        pred = self.classifier_new(x)

        return pred

