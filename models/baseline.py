"""
Baseline Siamese 2D CNN model.
"""

# Non-standard libraries
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from tqdm import tqdm

# Custom libraries
from utilities.kornia_augmentation import DataAugmentation


class SiamNet(pl.LightningModule):
    def __init__(self, augmentation: DataAugmentation = None,
                 adam=False, momentum=0.9, weight_decay=0.0005, dropout_rate=0.5, include_cov=False,
                 lr=0.005, output_dim=256, weighted_loss=0.5, model_hyperparams=None):
        """
        :param augmentation Sequential module for data augmentation
        :param model_hyperparams Alternative way to feed hyperparameters through dictionary
        """
        super(SiamNet, self).__init__()

        self.updated_naming = True

        # Update hyperparameters if given in dictionary
        if model_hyperparams is not None:
            self.save_hyperparameters(model_hyperparams)
        else:
            self.save_hyperparameters()

        # For image augmentation
        self.augmentation = augmentation

        # Define loss and metrics
        self.loss = torch.nn.NLLLoss(weight=torch.tensor((1 - self.hparams.weighted_loss, self.hparams.weighted_loss)))

        self._metrics = {}
        dsets = ['train', 'val', 'test0', 'test1', 'test2', 'test3', 'test4']
        for dset in dsets:
            exec(f"self.{dset}_acc = torchmetrics.Accuracy()")
            exec(f"self.{dset}_auroc = torchmetrics.AUROC(num_classes=1, average='micro')")
            exec(f"self.{dset}_auprc= torchmetrics.AveragePrecision(num_classes=1)")

        # CONVOLUTIONAL BLOCKS
        if self.updated_naming:
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
            self.fc8.add_module('drop8', nn.Dropout(p=self.hparams.dropout_rate))

            self.fc9 = nn.Sequential()
            self.fc9.add_module('fc9', nn.Linear(2 * 512, self.hparams.output_dim))
            self.fc9.add_module('relu9', nn.ReLU(inplace=True))
            self.fc9.add_module('drop9', nn.Dropout(p=self.hparams.dropout_rate))

            self.fc10 = nn.Sequential()
            self.fc10.add_module('fc10', nn.Linear(self.hparams.output_dim, 2))

            if self.hparams.include_cov:
                self.fc10.add_module('relu10', nn.ReLU(inplace=True))

                self.fc10b = nn.Sequential()
                self.fc10b.add_module('fc10b', nn.Linear(4, self.hparams.output_dim))
                self.fc10b.add_module('relu10b', nn.ReLU(inplace=True))

                self.fc10c = nn.Sequential()
                self.fc10c.add_module('fc10c', nn.Linear(self.hparams.output_dim, 2))
        else:
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

            self.fc7_new = nn.Sequential()
            self.fc7_new.add_module('fc7', nn.Linear(2 * 512, 256))
            self.fc7_new.add_module('relu7', nn.ReLU(inplace=True))

            self.classifier_new = nn.Sequential()
            self.classifier_new.add_module('fc8', nn.Linear(256, 2))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)

        if 'model_state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["model_state_dict"]
        # pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def configure_optimizers(self):
        if self.hparams.adam:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, data):
        """Images in batch input is of the form (B, V, 256, 256) where V=view (sagittal, transverse). Model expects
        image pixel intensities to be in [0, 255]."""
        x = data['img']
        if self.updated_naming:
            x = torch.div(x, 255)

        B, V, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            if self.updated_naming:
                z = self.conv1(z)
                z = self.conv2(z)
                z = self.conv3(z)
                z = self.conv4(z)
                z = self.conv5(z)
                z = self.conv6(z)
                z = self.conv7(z)
                z = z.view([B, 1, -1])
                z = self.fc8(z)
            else:
                z = self.conv(z)
                z = self.fc6(z)
                z = self.fc6b(z)
                z = z.view([B, 1, -1])
                z = self.fc6c(z)
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(B, -1)
        if self.updated_naming:
            x = self.fc9(x)
            x = self.fc10(x)
        else:
            x = self.fc7_new(x)
            x = self.classifier_new(x)

        if self.hparams.include_cov:
            age = data['Age_wks'].view(B, 1)
            side = data['Side_L'].view(B, 1)

            x = torch.cat((x, age, side), 1)
            x = self.fc10b(x)
            x = self.fc10c(x)

        return torch.log_softmax(x, dim=1)

    def training_step(self, train_batch, batch_idx):
        data_dict, y_true, id_ = train_batch

        if self.augmentation is not None:
            data_dict['img'] = self.augmentation(data_dict['img'])

        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        loss = self.loss(out, y_true)
        self.train_acc.update(y_pred, y_true)
        self.train_auroc.update(out[:, 1], y_true)
        self.train_auprc.update(out[:, 1], y_true)

        return loss

    def validation_step(self, val_batch, batch_idx):
        data_dict, y_true, id_ = val_batch
        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        loss = self.loss(out, y_true)
        self.val_acc.update(y_pred, y_true)
        self.val_auroc.update(out[:, 1], y_true)
        self.val_auprc.update(out[:, 1], y_true)

        return loss

    def test_step(self, test_batch, batch_idx, dataset_idx=0):
        data_dict, y_true, id_ = test_batch

        if self.augmentation is not None:
            print("Augmenting testing images")
            self.augmentation.set_mode('eval')
            data_dict['img'] = self.augmentation(data_dict['img'])

        out = self.forward(data_dict)
        y_pred = torch.argmax(out, dim=1)

        loss = self.loss(out, y_true)
        dset = f'test{dataset_idx}'

        exec(f'self.{dset}_acc.update(y_pred, y_true)')
        exec(f'self.{dset}_auroc.update(out[:, 1], y_true)')
        exec(f'self.{dset}_auprc.update(out[:, 1], y_true)')

        return loss

    def training_epoch_end(self, outputs):
        """Compute, log, and reset metrics for epoch."""
        loss = torch.stack([d['loss'] for d in outputs]).mean()
        acc = self.train_acc.compute()
        auroc = self.train_auroc.compute()
        auprc = self.train_auprc.compute()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_auroc', auroc)
        self.log('train_auprc', auprc, prog_bar=True)

        self.train_acc.reset()
        self.train_auroc.reset()
        self.train_auprc.reset()

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.tensor(validation_step_outputs).mean()
        acc = self.val_acc.compute()
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_auroc', auroc)
        self.log('val_auprc', auprc, prog_bar=True)

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

    def test_epoch_end(self, test_step_outputs):
        num_datasets = len(test_step_outputs) if len(test_step_outputs) <= 5 else 1
        for i in range(num_datasets):
            dset = f'test{i}'

            loss = torch.tensor(test_step_outputs[i]).mean()
            acc = eval(f'self.{dset}_acc.compute()')
            auroc = eval(f'self.{dset}_auroc.compute()')
            auprc = eval(f'self.{dset}_auprc.compute()')

            print(acc, auroc, auprc)

            self.log(f'{dset}_loss', loss)
            self.log(f'{dset}_acc', acc)
            self.log(f'{dset}_auroc', auroc)
            self.log(f'{dset}_auprc', auprc, prog_bar=True)

            exec(f'self.{dset}_acc.reset()')
            exec(f'self.{dset}_auroc.reset()')
            exec(f'self.{dset}_auprc.reset()')

    @torch.no_grad()
    def forward_embed(self, data):
        x = data['img']
        x = torch.div(x, 255)

        B, T, C, H = x.size()
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
            z = z.view([B, 1, -1])
            z = self.fc8(z)
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = x.view(B, -1)
        # x = self.fc9(x)

        return x.cpu().detach().numpy()

    @torch.no_grad()
    def extract_embeddings(self, dataloader):
        """Extract embeddings, and return labels and IDs for all items in inserted dataloader."""
        embed_lst = []
        y_list = []
        id_list = []

        self.eval()
        for batch_idx, (data_dict, y_true, id_) in tqdm(enumerate(dataloader)):
            out = self.forward_embed(data_dict)
            embed_lst.append(out)
            y_list.append(y_true.numpy())
            id_list.append(id_)

        embeds = np.concatenate(embed_lst, axis=0)
        labels = np.concatenate(y_list, axis=None)
        ids = np.concatenate(id_list, axis=None)

        return embeds, labels, ids

    @torch.no_grad()
    def extract_preds(self, dataloader):
        """Predict samples in dataloader. Return log-probability of positive prediction for every example."""
        preds = []
        y_list = []
        id_list = []

        self.eval()
        for batch_idx, (data_dict, y_true, id_) in tqdm(enumerate(dataloader)):
            out = self.forward(data_dict)
            preds.append(out[:, 1].cpu().detach().numpy())  # get probability for positive prediction
            y_list.append(y_true.numpy())
            id_list.append(id_)

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(y_list, axis=None)
        ids = np.concatenate(id_list, axis=None)

        return preds, labels, ids


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
