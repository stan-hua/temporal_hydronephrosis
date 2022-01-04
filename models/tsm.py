"""Baseline Siamese 2D CNN model with Temporal Shift Module (TSM).
"""

import numpy as np
import torch
from torch import nn

from models.baseline import SiamNet
from models.tsm_blocks import *


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetTSM(SiamNet):
    def __init__(self, classes=2, num_inputs=2, output_dim=128, cov_layers=False, device=None, dropout_rate=0.5):
        super().__init__(classes, num_inputs, output_dim, cov_layers, device, dropout_rate)

        self.conv.conv2_s1 = TemporalShift(self.conv.conv2_s1)
        self.conv.conv3_s1 = TemporalShift(self.conv.conv3_s1)
        self.conv.conv4_s1 = TemporalShift(self.conv.conv4_s1)
        self.conv.conv5_s1 = TemporalShift(self.conv.conv5_s1)
        self.fc6.fc6_s1 = TemporalShift(self.fc6.fc6_s1)
        self.fc6b.conv6b_s1 = TemporalShift(self.fc6b.conv6b_s1)

    def forward(self, x_t):
        """Accepts sequence of dual view images. Predicts label at each time step.

        ==Precondition==:
            - batch size is 1
            - without covariates and length, x_t shape is (1, time, dual_view, height, width)
        """
        if self.cov_layers:
            data, in_dict = data
            x_t, x_lengths = data
            x_t = x_t, in_dict
        else:
            x_t, x_lengths = data

        x = torch.squeeze(x_t, 0)
        T, C, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)
            z = self.conv(curr_x)
            z = self.fc6(z)
            z = self.fc6b(z)
            z = z.view([T, 1, -1])
            z = self.fc6c(z)
            z = z.view([T, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(T, -1)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(T, 1)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(T, 1)

            age = age.expand(-1, 512)
            side = side.expand(-1, 512)
            x = torch.cat((x, age, side), 1)

        x = self.fc7_new(x)
        x = self.fc8(x)
        pred = self.classifier_new(x)

        return pred
