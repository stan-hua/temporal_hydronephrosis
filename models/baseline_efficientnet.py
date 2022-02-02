"""
Baseline Siamese EfficientNet CNN.
"""

import pytorch_lightning as pl
import torch
import torchmetrics.functional
from torch import nn
from utilities.kornia_augmentation import DataAugmentation
from efficientnet_pytorch import EfficientNet


class SiameseEfficientNet(pl.LightningModule):
    def __init__(self, model_hyperparams=None, augmentation: DataAugmentation = None):
        """
        :param model_hyperparams: dictionary/namespace containing the following hyperparameters...
            lr: learning rate number
            batch_size: batch size number
            adam: boolean value. If true, use adam optimizer. Otherwise, SGD is used.
            include_cov: include covariate layers in input and forward pass
            dropout_rate: dropout rate for linear layers fc8, fc9
            weighted_loss: weight between (0, 1) to assign to positive class. Negative class receives 1 - weighted_loss.
            output_dim: dimensionality of features in layer before prediction layer
            stop_epoch: epoch at which to stop at
        """
        super().__init__()

        self.efficientnet = EfficientNet.from_name('efficientnet-b2', num_classes=2)

        # Save hyperparameters to checkpoint
        self.save_hyperparameters(model_hyperparams)

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

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(2816, 2))

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
        if self.hparams.adam:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)
        return optimizer

    def forward(self, data):
        """Images in batch input is of the form (B,V,H,W) where V=view (sagittal, transverse)"""
        x = data['img']
        x = x.transpose(0, 1)
        x_list = []
        for i in range(2):  # extract features for each US plane (sag, trv)
            z = torch.unsqueeze(x[i], 1)
            z = z.expand(-1, 3, -1, -1)
            z = self.efficientnet.extract_features(z)
            x_list.append(z)

        # Average features over each plane
        # x = torch.mean(torch.stack(x_list), 0)
        x = torch.cat(x_list, 1)

        # Pooling and final linear layer
        x = self.efficientnet._avg_pooling(x)
        if self.efficientnet._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.efficientnet._dropout(x)
            x = self.fc(x)

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

    def test_step(self, test_batch, batch_idx, dataset_idx):
        data_dict, y_true, id_ = test_batch
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
        for i in range(len(test_step_outputs)):
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


if __name__ == '__main__':
    import umap
    import numpy as np
    import torch
    from utilities.data_visualizer import plot_umap

    hyperparams = {'lr': 0.001, "batch_size": 16,
                   'adam': True,
                   'momentum': 0.9,
                   'weight_decay': 0.0005,
                   'include_cov': False,
                   'output_dim': 128,
                   'dropout_rate': 0,
                   'weighted_loss': 0.5,
                   'stop_epoch': 40}

    model = SiameseEfficientNet(hyperparams)
    n = 10
    data = {"img": torch.from_numpy(np.random.rand(n, 2, 260, 260)).type(torch.FloatTensor),
            "Age_wks": torch.from_numpy((np.random.rand(n) * 40).round()).type(torch.FloatTensor),
            "Side_L": torch.from_numpy(np.random.randint(0, 2, n)).type(torch.FloatTensor)}
    labels = np.random.randint(0, 2, n)
    output_ = model.forward(data)
