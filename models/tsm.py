"""
Baseline Siamese 2D CNN model, modified to convert convolutional layers into Temporal Shift Modules (TSM).

Due to variable sequence length, <n_segment> in TemporalShift block is assigned 1. This means that a portion is
shifted temporally for every channel.
"""

import numpy as np
import torch
from torch import nn

from models.baseline_pl import SiamNet
from models.tsm_blocks import *


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetTSM(SiamNet):
    def __init__(self, model_hyperparams):
        super().__init__(model_hyperparams)

        self.conv2.conv2_s1 = TemporalShift(self.conv2.conv2_s1, n_segment=1, inplace=True)
        self.conv3.conv3_s1 = TemporalShift(self.conv3.conv3_s1, n_segment=1, inplace=True)
        self.conv4.conv4_s1 = TemporalShift(self.conv4.conv4_s1, n_segment=1, inplace=True)
        self.conv5.conv5_s1 = TemporalShift(self.conv5.conv5_s1, n_segment=1, inplace=True)
        self.conv6.conv6_s1 = TemporalShift(self.conv6.conv6_s1, n_segment=1, inplace=True)
        self.conv7.conv7_s1 = TemporalShift(self.conv7.conv7_s1, n_segment=1, inplace=True)

    def forward(self, data):
        """Batch of images correspond to the images for one patient, where it is of the form (T,V,H,W).
        V refers to ultrasound view/plane (sagittal, transverse) and T refers to number of time points.
        """
        x = data['img'][0]

        T, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
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
            z = z.view([T, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(T, -1)
        x = self.fc9(x)
        x = self.fc10(x)

        if self.hparams.include_cov:
            age = data['Age_wks'].view(T, 1)
            side = data['Side_L'].view(T, 1)

            x = torch.cat((x, age, side), 1)
            x = self.fc10b(x)
            x = self.fc10c(x)

        # Average logits over time
        x = torch.mean(x, dim=0, keepdim=True)

        return torch.log_softmax(x, dim=1)
