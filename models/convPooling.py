"""Baseline Siamese 2D CNN model with Conv Pooling.
"""

import numpy as np
import torch
from torch import nn

from models.baselineSiamese import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetConvPooling(SiamNet):
    def forward(self, x_t):
        """Accepts sequence of dual view images. Extracts penultimate layer embeddings for each dual view, then performs
        max pooling over time.
        """
        t_embeddings = []

        for x in x_t:
            if torch.mean(x) == 0:
                continue

            if self.cov_layers:
                in_dict = x
                x = in_dict['img']

            # x = x.unsqueeze(0)
            if self.num_inputs == 1:
                x = x.unsqueeze(1)
            #   B, C, H = x.size()
            # else:
            B, T, C, H = x.size()
            x = x.transpose(0, 1)
            x_list = []
            for i in range(self.num_inputs):
                if self.num_inputs == 1:
                    curr_x = torch.unsqueeze(x[i], 1)
                else:
                    curr_x = torch.unsqueeze(x[i], 1)
                # if self.num_inputs == 1:
                #   curr_x = curr_x.expand(-1, 3, -1)
                # else:
                curr_x = curr_x.expand(-1, 3, -1, -1)
                if torch.cuda.is_available():
                    input = torch.cuda.FloatTensor(curr_x.to(self.device))
                else:
                    input = torch.FloatTensor(curr_x.to(self.device))
                z = self.conv(input)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])
                z = self.fc6c(z)
                z = z.view([B, 1, -1])
                x_list.append(z)

            x = torch.cat(x_list, 1)
            # x = torch.sum(x, 1)
            x = self.fc7_new(x.view(B, -1))
            t_embeddings.append(x)

        x, _ = torch.max(torch.stack(t_embeddings), dim=1)
        pred = self.classifier_new(x)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Age: ")
            # print(age)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(device).view(B, 1)
            # print("Side: ")
            # print(side)
            mid_in = torch.cat((pred, age, side), 1)

            x = self.add_covs1(mid_in)
            pred = self.add_covs2(x)

        return pred

