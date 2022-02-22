"""
Baseline Siamese 2D CNN model, modified to convert convolutional layers into Temporal Shift Modules (TSM).

Due to variable sequence length, <n_segment> in TemporalShift block is assigned 1. This means that a portion is
shifted temporally for every channel.
"""

from models.baseline import SiamNet
from models.tsm_blocks import *
from utilities.kornia_augmentation import DataAugmentation


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNetTSM(SiamNet):
    def __init__(self, augmentation: DataAugmentation = None,
                 adam=False, momentum=0.9, weight_decay=0.0005, dropout_rate=0.5, include_cov=False,
                 lr=0.005, output_dim=256, weighted_loss=0.5, model_hyperparams=None):
        self.save_hyperparameters()
        if model_hyperparams is not None:
            super().__init__(augmentation=augmentation, model_hyperparams=model_hyperparams)
        else:
            super().__init__(augmentation=augmentation, model_hyperparams=self.hparams)

        self.shift = TemporalShift(inplace=True)

    def forward(self, data):
        """Batch of images correspond to the images for one patient, where it is of the form (T,V,H,W).
        V refers to ultrasound view/plane (sagittal, transverse) and T refers to number of time points.

        Perform conv. pooling to pool temporal features.
        """
        x = data['img']
        x = torch.div(x, 255)

        if len(x.size()) == 5:
            x = x[0]

        T, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            z = self.conv1(z)

            z = self.shift(z)
            z = self.conv2(z)

            z = self.shift(z)
            z = self.conv3(z)

            z = self.shift(z)
            z = self.conv4(z)

            z = self.shift(z)
            z = self.conv5(z)

            z = self.shift(z)
            z = self.conv6(z)

            z = self.shift(z)
            z = self.conv7(z)

            z = z.view([T, 1, -1])
            z = self.fc8(z)
            z = z.view([T, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(T, -1)

        # Get features at last visit (time point)
        x = x[-1:, :]

        x = self.fc9(x)
        x = self.fc10(x)

        return torch.log_softmax(x, dim=1)

    def alt_forward_avg(self, data):
        """Batch of images correspond to the images for one patient, where it is of the form (T,V,H,W).
        V refers to ultrasound view/plane (sagittal, transverse) and T refers to number of time points.

        Perform average prediction to pool temporal features.
        """
        x = data['img']
        x = torch.div(x, 255)

        if len(x.size()) == 5:
            x = x[0]

        T, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            z = self.conv1(z)

            z = self.shift(z)
            z = self.conv2(z)

            z = self.shift(z)
            z = self.conv3(z)

            z = self.shift(z)
            z = self.conv4(z)

            z = self.shift(z)
            z = self.conv5(z)

            z = self.shift(z)
            z = self.conv6(z)

            z = self.shift(z)
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

    def alt_forward_conv(self, data):
        """Batch of images correspond to the images for one patient, where it is of the form (T,V,H,W).
        V refers to ultrasound view/plane (sagittal, transverse) and T refers to number of time points.

        Perform conv. pooling to pool temporal features.
        """
        x = data['img']
        x = torch.div(x, 255)

        if len(x.size()) == 5:
            x = x[0]

        T, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            z = self.conv1(z)

            z = self.shift(z)
            z = self.conv2(z)

            z = self.shift(z)
            z = self.conv3(z)

            z = self.shift(z)
            z = self.conv4(z)

            z = self.shift(z)
            z = self.conv5(z)

            z = self.shift(z)
            z = self.conv6(z)

            z = self.shift(z)
            z = self.conv7(z)

            z = z.view([T, 1, -1])
            z = self.fc8(z)
            z = z.view([T, 1, -1])
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

        T, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            z = self.conv1(z)

            z = self.shift(z)
            z = self.conv2(z)

            z = self.shift(z)
            z = self.conv3(z)

            z = self.shift(z)
            z = self.conv4(z)

            z = self.shift(z)
            z = self.conv5(z)

            z = self.shift(z)
            z = self.conv6(z)

            z = self.shift(z)
            z = self.conv7(z)

            z = z.view([T, 1, -1])
            z = self.fc8(z)
            z = z.view([T, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(T, -1)
        x = self.fc9(x)
        x = self.fc10(x)

        # Average logits over time
        x = torch.mean(x, dim=0, keepdim=True)

        return x.cpu().detach().numpy()
