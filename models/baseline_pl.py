"""Baseline Siamese 2D CNN model.
"""

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional


# noinspection PyTypeChecker,PyUnboundLocalVariable
class SiamNet(pl.LightningModule):
    def __init__(self, classes=2, num_inputs=2, output_dim=128, cov_layers=False, dropout_rate=0.5,
                 args=None, hyperparameters=None):
        super(SiamNet, self).__init__()

        self.args = args

        # Model-specific arguments
        self.cov_layers = cov_layers
        self.output_dim = output_dim
        self.num_inputs = num_inputs
        self.hyperparameters = hyperparameters

        # Save hyperparameters to checkpoint
        self.save_hyperparameters()

        # CONVOLUTIONAL BLOCKS
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0, bias=False))
        self.conv1.add_module('batch1_s1', nn.BatchNorm2d(96))
        self.conv1.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv1.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv1.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2, bias=False))
        self.conv2.add_module('batch2_s1', nn.BatchNorm2d(256))
        self.conv2.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv2.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False))
        self.conv3.add_module('batch3_s1', nn.BatchNorm2d(384))
        self.conv3.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False))
        self.conv4.add_module('batch4_s1', nn.BatchNorm2d(384))
        self.conv4.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential()
        self.conv5.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False))
        self.conv5.add_module('batch5_s1', nn.BatchNorm2d(256))
        self.conv5.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv5.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv6 = nn.Sequential()
        self.conv6.add_module('conv6_s1', nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1, bias=False))
        self.conv6.add_module('batch6_s1', nn.BatchNorm2d(1024))
        self.conv6.add_module('relu6_s1', nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential()
        self.conv7.add_module('conv7_s1', nn.Conv2d(1024, 256, kernel_size=3, stride=2, bias=False))
        self.conv7.add_module('batch7_s1', nn.BatchNorm2d(256))
        self.conv7.add_module('relu7_s1', nn.ReLU(inplace=True))
        self.conv7.add_module('pool7_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        # DENSE LAYERS
        self.fc8 = nn.Sequential()
        self.fc8.add_module('fc8', nn.Linear(256 * 3 * 3, 512))
        self.fc8.add_module('relu8', nn.ReLU(inplace=True))
        self.fc8.add_module('drop8', nn.Dropout(p=dropout_rate))

        self.fc9 = nn.Sequential()
        self.fc9.add_module('fc9', nn.Linear(self.num_inputs * 512, self.output_dim))
        self.fc9.add_module('relu9', nn.ReLU(inplace=True))
        self.fc9.add_module('drop9', nn.Dropout(p=dropout_rate))

        # self.fc10 = nn.Sequential()
        # self.fc10.add_module('fc10', nn.Linear(512, self.output_dim))
        # self.fc10.add_module('relu10', nn.ReLU(inplace=True))
        # self.fc10.add_module('drop10', nn.Dropout(p=dropout_rate))

        self.fc10 = nn.Sequential()
        self.fc10.add_module('fc10', nn.Linear(self.output_dim, classes))

        if self.cov_layers:
            self.fc10.add_module('relu10', nn.ReLU(inplace=True))

            self.fc10b = nn.Sequential()
            self.fc10b.add_module('fc10b', nn.Linear(classes + 2, self.output_dim))
            self.fc10b.add_module('relu10b', nn.ReLU(inplace=True))

            self.fc10c = nn.Sequential()
            self.fc10c.add_module('fc10c', nn.Linear(classes + 126, classes))

        self.loss = torch.nn.NLLLoss(weight=torch.tensor((0.12, 0.88)))      # weight=torch.tensor((0.15, 0.85))
        # self.loss = torch.nn.CrossEntropyLoss()

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

    def configure_optimizers(self):
        if self.args.adam:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters['lr'],
                                         weight_decay=self.hyperparameters['weight_decay'])
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hyperparameters['lr'],
                                        momentum=self.hyperparameters['momentum'],
                                        weight_decay=self.hyperparameters['weight_decay'])
        return optimizer

    def forward(self, data):
        x = data['img']

        if self.cov_layers:
            cov_dict = data['cov']

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
        x = self.fc9(x)
        x = self.fc10(x)

        if self.cov_layers:
            age = cov_dict['Age_wks'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)
            side = cov_dict['Side_L'].type(torch.FloatTensor).to(self.device, non_blocking=True).view(B, 1)

            x = torch.cat((x, age, side), 1)
            x = self.fc10b(x)
            x = self.fc10c(x)

        return torch.log_softmax(x, dim=1)

    def training_step(self, train_batch, batch_idx):
        data, y_true, cov = train_batch
        data_dict = {}
        if self.args.standardize_seq_length:
            data_dict['img'] = data[0]
            data_dict['length'] = data[1]
        else:
            data_dict['img'] = data

        data_dict['cov'] = cov if self.args.include_cov else None

        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        loss = self.loss(out, y_true)
        acc = torchmetrics.functional.accuracy(y_pred, y_true)
        # auroc = torchmetrics.functional.auroc(y_prob, y_true, num_classes=2, average="macro")
        # auprc = torchmetrics.functional.average_precision(y_prob, y_true, num_classes=2)
        auroc = torchmetrics.functional.auroc(out[:, 1], y_true, num_classes=1, average="micro")
        auprc = torchmetrics.functional.average_precision(out[:, 1], y_true, num_classes=1)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        self.log('train_auprc', auprc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        data, y_true, cov = val_batch

        data_dict = {}
        if self.args.standardize_seq_length:
            data_dict['img'] = data[0]
            data_dict['length'] = data[1]
        else:
            data_dict['img'] = data

        data_dict['cov'] = cov if self.args.include_cov else None

        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        print(out)
        print(y_pred)
        print(y_true)

        loss = self.loss(out, y_true)
        acc = torchmetrics.functional.accuracy(y_pred, y_true)
        # auroc = torchmetrics.functional.auroc(y_prob, y_true, num_classes=2, average="macro")
        # auprc = torchmetrics.functional.average_precision(y_prob, y_true, num_classes=2)
        auroc = torchmetrics.functional.auroc(out[:, 1], y_true, num_classes=1, average="micro")
        auprc = torchmetrics.functional.average_precision(out[:, 1], y_true, num_classes=1)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_auroc', auroc, on_step=False, on_epoch=True)
        self.log('val_auprc', auprc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        data, y_true, cov = test_batch

        data_dict = {}
        if self.args.standardize_seq_length:
            data_dict['img'] = data[0]
            data_dict['length'] = data[1]
        else:
            data_dict['img'] = data

        data_dict['cov'] = cov if self.args.include_cov else None

        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        loss = self.loss(out, y_true)
        acc = torchmetrics.functional.accuracy(y_pred, y_true)
        # auroc = torchmetrics.functional.auroc(y_prob, y_true, num_classes=2, average="macro")
        # auprc = torchmetrics.functional.average_precision(y_prob, y_true, num_classes=2)
        auroc = torchmetrics.functional.auroc(out[:, 1], y_true, num_classes=1, average="micro")
        auprc = torchmetrics.functional.average_precision(out[:, 1], y_true, num_classes=1)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_auroc', auroc, on_step=False, on_epoch=True)
        self.log('test_auprc', auprc, on_step=False, on_epoch=True)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        data, target, cov = batch

        if self.args.standardize_seq_length:
            return (data[0].to(device, non_blocking=True), torch.from_numpy(data[1])), \
                   target.type(torch.LongTensor).to(device), self.to_device(cov, device)
        else:
            return data.to(device, non_blocking=True), target.type(torch.LongTensor).to(device), \
                   self.to_device(cov, device)

    def to_device(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_obj[k] = self.to_device(v, device)
            return new_obj
        else:
            raise NotImplementedError("Only of type tensor and dictionary are implemented!")


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
