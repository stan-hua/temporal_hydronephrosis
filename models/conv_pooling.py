"""
Baseline Siamese 2D CNN model, modified to perform conv. (max) pooling of convolutional features over time.
"""

import numpy as np
import torch

from models.baseline import SiamNet


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetConvPooling(SiamNet):
    def forward(self, data):
        """Accepts sequence of dual view images for one patient. Extracts penultimate layer embeddings, then performs
        max pooling over time.

        ==Precondition==:
            - data is fed in batch sizes of 1
            - data['img'] is of the form (1, T, C, H, W) or (T, C, H, W), where T=time, V=view (sagittal, transverse)
        """
        x = data['img']
        x = torch.div(x, 255)

        if len(x.size()) == 5:
            x = x[0]

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
        x = x.view(T, -1)

        # Max pooling over time
        x, _ = torch.max(x, dim=0, keepdim=True)

        x = self.fc9(x)
        x = self.fc10(x)

        return torch.log_softmax(x, dim=1)


    @torch.no_grad()
    def forward_embed(self, data):
        x = data['img']
        x = torch.div(x, 255)

        if len(x.size()) == 5:
            x = x[0]

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
        x = x.view(T, -1)

        # Max pooling over time
        x, _ = torch.max(x, dim=0, keepdim=True)

        # x = self.fc9(x)
        # x = self.fc10(x)

        return x.cpu().detach().numpy()
