"""Baseline Siamese 2D CNN model.
"""

import torch
from torch import nn


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNet(nn.Module):
    def __init__(self, classes=2, num_inputs=2, output_dim=128, cov_layers=False, device=None, dropout_rate=0.5):
        self.device = device
        super(SiamNet, self).__init__()

        self.cov_layers = cov_layers
        self.output_dim = output_dim
        self.num_inputs = num_inputs

        # CONVOLUTIONAL BLOCKS
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.fc6b = nn.Sequential()
        self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # DENSE LAYERS
        self.fc6c = nn.Sequential()
        self.fc6c.add_module('fc7', nn.Linear(256 * 3 * 3, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        # modified
        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear((self.num_inputs + (2 if self.cov_layers else 0)) * 512, 512))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        # modified
        self.fc8 = nn.Sequential()
        self.fc8.add_module('fc8', nn.Linear(512, self.output_dim))
        self.fc8.add_module('relu8', nn.ReLU(inplace=True))
        # self.fc8.add_module('drop8', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc9', nn.Linear(self.output_dim, classes))      # changed fc8 -> fc9

        # if self.cov_layers:
        #     # self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))
        #
        #     self.add_covs1 = nn.Sequential()
        #     self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
        #     self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))
        #
        #     self.add_covs2 = nn.Sequential()
        #     self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)

        if 'model_state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["model_state_dict"]
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        if self.cov_layers:
            x, in_dict = x

        if self.num_inputs == 1:
            x = x.unsqueeze(1)

        B, T, C, H = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.num_inputs):
            curr_x = torch.unsqueeze(x[i], 1)
            curr_x = curr_x.expand(-1, 3, -1, -1)

            if torch.cuda.is_available():
                input_ = torch.cuda.FloatTensor(curr_x.to(device))
            else:
                input_ = torch.FloatTensor(curr_x.to(device))

            z = self.conv(input_)
            z = self.fc6(z)
            z = self.fc6b(z)
            z = z.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(B, -1)

        if self.cov_layers:
            age = in_dict['Age_wks'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)
            side = in_dict['Side_L'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)

            age = age.expand(-1, 512)
            side = side.expand(-1, 512)
            x = torch.cat((x, age, side), 1)

        x = self.fc7_new(x)
        x = self.fc8(x)
        pred = self.classifier_new(x)

        return pred


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)

        x = x.div(div)
        return x
