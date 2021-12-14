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
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_s1', nn.Conv2d(1, 96, kernel_size=11, stride=2, padding=0))
        self.conv1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv1.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv1.add_module('pool1_s2', nn.MaxPool2d(kernel_size=2, stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv2.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        # self.conv2.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv2.add_module('conv2b', nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1', nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv3.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv3.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv4.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv4.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv5.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv5.add_module('pool5_s1', nn.MaxPool2d(kernel_size=2, stride=2))

        # *************************** changed layers *********************** #
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1))
        self.fc6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('pool6_s1', nn.MaxPool2d(kernel_size=2, stride=1))

        # self.fc6b = nn.Sequential()
        # self.fc6b.add_module('conv6b_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2))
        # self.fc6b.add_module('batch6b_s1', nn.BatchNorm2d(256))
        # self.fc6b.add_module('relu6_s1', nn.ReLU(inplace=True))
        # self.fc6b.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # TODO: Rename this
        self.uconnect1 = nn.Sequential()
        self.uconnect1.add_module('conv', nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.uconnect1.add_module('batch', nn.BatchNorm2d(256))
        self.uconnect1.add_module('relu', nn.ReLU(inplace=True))
        self.uconnect1.add_module('pool6b_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # FC (cont.)
        self.fc6c = nn.Sequential()
        self.fc6c.add_module('fc7', nn.Linear(256 * 7 * 7, 512))
        self.fc6c.add_module('relu7', nn.ReLU(inplace=True))
        self.fc6c.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.fc7_new = nn.Sequential()
        self.fc7_new.add_module('fc7', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))
        # self.fc7_new.add_module('drop7', nn.Dropout(p=dropout_rate))

        self.classifier_new = nn.Sequential()
        self.classifier_new.add_module('fc8', nn.Linear(self.output_dim, classes))

        if self.cov_layers:
            self.classifier_new.add_module('relu8', nn.ReLU(inplace=True))

            self.add_covs1 = nn.Sequential()
            self.add_covs1.add_module('fc9', nn.Linear(classes + 2, classes + 126))
            self.add_covs1.add_module('relu9', nn.ReLU(inplace=True))

            self.add_covs2 = nn.Sequential()
            self.add_covs2.add_module('fc10', nn.Linear(classes + 126, classes))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)["model_state_dict"]
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # print([k for k, v in list(pretrained_dict.items())])

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
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

            z = unet1.view([B, 1, -1])
            z = self.fc6c(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc7_new(x.view(B, -1))
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

