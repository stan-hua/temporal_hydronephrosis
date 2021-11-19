import importlib.machinery
from collections import defaultdict
from datetime import datetime
import os
import sys
import argparse
import json
import warnings

from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable

from models.baselineSiamese import SiamNet
# from models.convPooling import SiamNet

warnings.filterwarnings('ignore')

SEED = 42
debug = False
model_name = "Siamese_Baseline"

results_dir = "C:\\Users\\Stanley Hua\\projects\\temporal_hydronephrosis\\results\\"

# Paths to save results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
curr_results_dir = f"{results_dir}{model_name}_{timestamp}/"
training_info_path = f"{curr_results_dir}info.csv"
auc_path = f"{curr_results_dir}auc.json"
results_summary_path = f"{curr_results_dir}history.csv"

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)


# Data-related Functions
class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, cov):
        # self.X = [torch.tensor(e, requires_grad=True).float() for e in X]
        self.X = [torch.from_numpy(e).float() for e in X]
        # self.X = torch.from_numpy(X).float()
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]
        return imgs, target, cov

    def __len__(self):
        return len(self.X)


def prepare_data_into_sequences(X_train, y_train, cov_train, X_test, y_test, cov_test, single_visit):
    """Prepare data into sequences of (pairs of images)"""

    def sort_data(t_x, t_y, t_cov):
        cov_train, X_train, y_train = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return X_train, y_train, cov_train

    def group(t_x, t_y, t_cov):
        """Group images according to patient ID"""
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            # split data per kidney e.g 5.0Left, 5.0Right, 6.0Left, ...
            id_ = t_cov[i].split("_")[0] + t_cov[i].split("_")[4]
            # id = t_cov[i].split("_")[0] # split only on id e.g. 5.0, 6.0, 7.0, ...
            x[id_].append(t_x[i])
            y[id_].append(t_y[i])
            cov[id_].append(t_cov[i])
        # convert to np array
        organized_X_train = np.asarray([np.asarray(e) for e in list(x.values())])
        return organized_X_train, np.asarray(list(y.values())), np.asarray(list(cov.values()))

    def get_only_last_visits(t_x, t_y, t_cov):
        """Slice data to get only latest n visits."""
        x, y, cov = [], [], []
        for i, e in enumerate(t_x):
            curr_x = e[-1:]
            curr_x = curr_x.transpose((1, 0, 2, 3))
            curr_x = curr_x.squeeze()
            x.append(curr_x)
            y.append(t_y[i][-1])
            cov.append(t_cov[i][-1])
        return np.asarray(x, dtype=np.float64), y, cov

    X_train, y_train, cov_train = sort_data(X_train, y_train, cov_train)
    X_test, y_test, cov_test = sort_data(X_test, y_test, cov_test)

    if not single_visit:    # group images by patient ID
        X_train, y_train, cov_train = group(X_train, y_train, cov_train)
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)

    if single_visit:        # only test on last visit
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)
        X_test, y_test, cov_test = get_only_last_visits(X_test, y_test, cov_test)

    return X_train, y_train, cov_train, X_test, y_test, cov_test


def make_validation_set(X_train, y_train, cov_train):
    """HELPER FUNCTION. Split training data into training and validation splits"""

    def split(train, split=0.2):
        """Save last person as a validation test?"""
        return train[:-int(len(train) * split)], train[-int(len(train) * split):]
        # return train[:-1], train[-1:]

    X_train, val_X = split(X_train)
    y_train, y_val = split(y_train)
    cov_train, val_cov = split(cov_train)
    return X_train, y_train, cov_train, val_X, y_val, val_cov


# Defining data/model parameters
def parseArgs():
    """When running script, parse user input for argument values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.005, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--output_dim", default=128, type=int, help="output dim for last linear layer")
    parser.add_argument("--git_dir", default="C:/Users/Stanley Hua/projects/")
    parser.add_argument("--stop_epoch", default=100, type=int,
                        help="If not running cross validation, which epoch to finish with")
    parser.add_argument("--cv_stop_epoch", default=18, type=int, help="get a pth file from a specific epoch")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")
    parser.add_argument('--cv', action='store_true', help="Flag to run cross validation")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument('--vgg', action='store_true', help="Run VGG16 architecture, not using this flag runs ResNet")
    parser.add_argument('--vgg_bn', action='store_true', help="Run VGG16 batch norm architecture")
    parser.add_argument('--densenet', action='store_true', help="Run DenseNet")
    parser.add_argument('--resnet18', action='store_true',
                        help="Run ResNet18 architecture, not using this flag runs ResNet16")
    parser.add_argument('--resnet50', action='store_true',
                        help="Run ResNet50 architecture, not using this flag runs ResNet50")
    parser.add_argument('--pretrained', action="store_true",
                        help="Use pretrained model with cross validation if cv requested")
    return parser.parse_args()


def modifyArgs(args):
    """Overwrite user-inputted model parameters.
    """
    args.lr = 0.001
    args.batch_size = 64
    args.stop_epoch = 100

    args.single_visit = True
    args.baseline = True

    # Save weights every x epochs
    args.save_frequency = 10

    # Include testing set?
    args.include_testing = True


def choose_model(args):
    num_inputs = 1 if args.view != "siamese" else 2
    model_pretrain = args.pretrained if args.cv else False

    if args.baseline:
        return SiamNet().to(device)


# Training
def train(args, X_train, y_train, cov_train, X_test, y_test, cov_test, val_X, y_val, val_cov):
    net = choose_model(args)
    hyperparams = {'lr': args.lr, 'momentum': args.momentum,
                   'weight_decay': args.weight_decay, 'train/test_split': args.split,
                   'num_epochs': args.epochs, 'stop_epoch': args.stop_epoch
                   }
    if args.adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                     weight_decay=hyperparams['weight_decay'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                    weight_decay=hyperparams['weight_decay'])
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    # Be careful to not shuffle order of image seq within a patient
    X_train, y_train, cov_train = shuffle(X_train, y_train, cov_train, random_state=SEED)

    if debug:
        X_train, y_train, cov_train = X_train[1:5], y_train[1:5], cov_train[1:5]
        X_test, y_test, cov_test = X_test[1:5], y_test[1:5], cov_test[1:5]
        if val_X:
            val_X, y_val, val_cov = val_X[1:5], y_val[1:5], val_cov[1:5]

    training_set = KidneyDataset(X_train, y_train, cov_train)
    test_set = KidneyDataset(X_test, y_test, cov_test)
    val_set = KidneyDataset(val_X, y_val, val_cov)

    training_generator = DataLoader(training_set, **params)
    test_generator = DataLoader(test_set, **params)
    val_generator = DataLoader(val_set, **params)
    print("Dataset generated")

    if debug:
        # prevents concurrent processing, to allow for stepping debug
        training_generator.num_workers = 0
        test_generator.num_workers = 0
        val_generator.num_workers = 0

    # Results accumulator
    res = Results(y_train, y_val, y_test, args)

    for epoch in range(args.stop_epoch + 1):
        # Reset results accumulator
        res.reset()

        print(f"Epoch {epoch}/{args.stop_epoch}")
        # Training
        net.train()
        for batch_idx, (data, target, cov) in enumerate(training_generator):
            optimizer.zero_grad()
            output = net(data.to(device))

            target = torch.tensor(target)  # change target to: tensor([0,1,0,1]) from: [tensor([0]), tensor([1]), tensor([0]), tensor([1])]
            target = Variable(target.type(torch.LongTensor), requires_grad=False).to(device)

            loss = F.cross_entropy(output, target)
            res.loss_accum_train += loss.item() * len(target)
            loss.backward()
            optimizer.step()

            output_softmax = softmax(output)
            pred_prob = output_softmax[:, 1]
            pred_label = torch.argmax(output_softmax, dim=1)

            res.accurate_pred_train += torch.sum(torch.argmax(output, dim=1) == target).cpu()
            res.counter_train += len(target)

            assert len(pred_prob) == len(target)
            assert len(pred_label) == len(target)
            res.all_pred_prob_train.append(pred_prob)
            res.all_pred_label_train.append(pred_label)
            res.all_targets_train.append(target)
            res.all_patient_ID_train.append(cov)

        # Validation Set
        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target, cov) in enumerate(val_generator):
                net.zero_grad()
                optimizer.zero_grad()
                output = net(data.to(device))
                target = torch.tensor(target)
                target = target.type(torch.LongTensor).to(device)
                loss = F.cross_entropy(output, target)
                res.loss_accum_val += loss.item() * len(target)
                res.counter_val += len(target)
                output_softmax = softmax(output)
                res.accurate_pred_val += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                pred_prob = output_softmax[:, 1]
                pred_label = torch.argmax(output, dim=1)
                assert len(pred_prob) == len(target)
                assert len(pred_label) == len(target)
                res.all_pred_prob_val.append(pred_prob)
                res.all_pred_label_val.append(pred_label)
                res.all_targets_val.append(target)
                res.all_patient_ID_val.append(cov)

        # Save results every epoch
        res.process_results(epoch, ["train", "val"], y_train, y_val, y_test)

        # Save model and optimizer weights
        if epoch % args.save_frequency == 0:
            model_path = f"{curr_results_dir}/model-epoch_{epoch}.pth"
            # optimizer_path = f"{curr_results_dir}/optimizer-epoch_{epoch}.pth"

            torch.save(net.state_dict(), model_path)
            # torch.save(optimizer.state_dict(), optimizer_path)

        # Save hyperparameters if not already saved
        if not os.path.exists(training_info_path):
            df_info = pd.DataFrame(hyperparams, index=[0])
            df_info.to_csv(training_info_path, index=False)

        # Testing Set
        if args.include_testing:
            net.eval()
            with torch.no_grad():
                for batch_idx, (data, target, cov) in enumerate(test_generator):
                    net.zero_grad()
                    optimizer.zero_grad()
                    output = net(data.to(device))
                    target = torch.tensor(target)
                    target = target.type(torch.LongTensor).to(device)
                    loss = F.cross_entropy(output, target)
                    res.loss_accum_test += loss.item() * len(target)
                    res.counter_test += len(target)
                    output_softmax = softmax(output)
                    res.accurate_pred_test += torch.sum(torch.argmax(output, dim=1) == target).cpu()
                    pred_prob = output_softmax[:, 1]
                    pred_label = torch.argmax(output, dim=1)
                    assert len(pred_prob) == len(target)
                    assert len(pred_label) == len(target)
                    res.all_pred_prob_test.append(pred_prob)
                    res.all_pred_label_test.append(pred_label)
                    res.all_targets_test.append(target)
                    res.all_patient_ID_test.append(cov)

            # Save test set results
            res.process_results(epoch, ["test"], y_train, y_val, y_test)


# Results Accumulator
class Results:
    """Stores results from model training and evaluation (on validation & testing set) per epoch.
    Also used to save model weights.
    """

    def __init__(self, y_train, y_val, y_test, args=None):
        self.epoch = -1
        self.dsets = []
        self.args = args

        self.all_targets_train, self.all_targets_val, self.all_targets_test = [], [], []

        self.accurate_pred_train, self.accurate_pred_val, self.accurate_pred_test = 0, 0, 0
        self.loss_accum_train, self.loss_accum_val, self.loss_accum_test = 0, 0, 0

        self.all_pred_prob_train, self.all_pred_prob_val, self.all_pred_prob_test = [], [], []
        self.all_pred_label_train, self.all_pred_label_val, self.all_pred_label_test = [], [], []

        self.all_patient_ID_train, self.all_patient_ID_val, self.all_patient_ID_test = [], [], []

        self.counter_train, self.counter_val, self.counter_test = 0, 0, 0
        # TODO: do something when it's not a sequence
        try:
            self.totalTrainItems, self.totalValItems, self.totalTestItems = (sum(len(seq) for seq in dset) for dset in
                                                                             [y_train, y_val, y_test])
        except:
            self.totalTrainItems, self.totalValItems, self.totalTestItems = [len(y_train), len(y_val), len(y_test)]

        self.all_targets_train_tensor, self.all_targets_val_tensor, self.all_targets_test_tensor = None, None, None
        self.all_pred_prob_train_tensor, self.all_pred_prob_val_tensor, self.all_pred_prob_test_tensor = None, None, None
        self.all_pred_label_train_tensor, self.all_pred_label_val_tensor, self.all_pred_label_test_tensor = None, None, None

    def reset(self):
        """Reset stored variables for next epoch."""
        self.all_targets_train, self.all_targets_val, self.all_targets_test = [], [], []

        self.accurate_pred_train, self.accurate_pred_val, self.accurate_pred_test = 0, 0, 0
        self.loss_accum_train, self.loss_accum_val, self.loss_accum_test = 0, 0, 0

        self.all_pred_prob_train, self.all_pred_prob_val, self.all_pred_prob_test = [], [], []
        self.all_pred_label_train, self.all_pred_label_val, self.all_pred_label_test = [], [], []

        self.all_patient_ID_train, self.all_patient_ID_val, self.all_patient_ID_test = [], [], []

        self.counter_train, self.counter_val, self.counter_test = 0, 0, 0

        self.all_targets_train_tensor, self.all_targets_val_tensor, self.all_targets_test_tensor = None, None, None
        self.all_pred_prob_train_tensor, self.all_pred_prob_val_tensor, self.all_pred_prob_test_tensor = None, None, None
        self.all_pred_label_train_tensor, self.all_pred_label_val_tensor, self.all_pred_label_test_tensor = None, None, None

    def concat_results(self):
        """HELPER FUNCTION. Concatenates targets, predicted label and prediction probabilities."""
        if "train" in self.dsets:
            self.all_targets_train_tensor = torch.cat(self.all_targets_train)
            self.all_pred_prob_train_tensor = torch.cat(self.all_pred_prob_train)
            self.all_pred_label_train_tensor = torch.cat(self.all_pred_label_train)

        if "val" in self.dsets:
            self.all_targets_val_tensor = torch.cat(self.all_targets_val)
            self.all_pred_prob_val_tensor = torch.cat(self.all_pred_prob_val)
            self.all_pred_label_val_tensor = torch.cat(self.all_pred_label_val)

        if "test" in self.dsets:
            self.all_targets_test_tensor = torch.cat(self.all_targets_test)
            self.all_pred_prob_test_tensor = torch.cat(self.all_pred_prob_test)
            self.all_pred_label_test_tensor = torch.cat(self.all_pred_label_test)

    def verify_length(self, y_train, y_val, y_test):
        """Verifies that the lengths of sequences in training/val/test set match predictions and labels."""
        if "train" in self.dsets:
            assert len(self.all_pred_prob_train_tensor) == self.totalTrainItems
            assert len(self.all_pred_label_train_tensor) == self.totalTrainItems
            assert len(self.all_targets_train_tensor) == self.totalTrainItems

            if not self.args.single_visit:
                assert len(self.all_patient_ID_train) == len(y_train)

        if "val" in self.dsets:
            assert len(self.all_pred_prob_val_tensor) == self.totalValItems
            assert len(self.all_pred_label_val_tensor) == self.totalValItems
            assert len(self.all_targets_val_tensor) == self.totalValItems

            if not self.args.single_visit:
                assert len(self.all_patient_ID_val) == len(y_val)

        if "test" in self.dsets:
            assert len(self.all_pred_prob_test_tensor) == self.totalTestItems
            assert len(self.all_pred_label_test_tensor) == self.totalTestItems
            assert len(self.all_targets_test_tensor) == self.totalTestItems

            if not self.args.single_visit:
                assert len(self.all_patient_ID_test) == len(y_test)

    def get_dset_results(self, dset):
        """HELPER FUNCTION. Return results for specified dset (train/val/test)."""
        if dset == "train":
            y_score = self.all_pred_prob_train_tensor.cpu().detach().numpy()
            y_true = self.all_targets_train_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_train_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_train) / self.counter_train
            loss = self.loss_accum_train / self.counter_train
        elif dset == "val":
            y_score = self.all_pred_prob_val_tensor.cpu().detach().numpy()
            y_true = self.all_targets_val_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_val_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_val) / self.counter_val
            loss = self.loss_accum_val / self.counter_val
        else:
            y_score = self.all_pred_prob_test_tensor.cpu().detach().numpy()
            y_true = self.all_targets_test_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_test_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_test) / self.counter_test
            loss = self.loss_accum_test / self.counter_test

        return y_score, y_true, y_pred, acc, loss

    def preprocess_results(self):
        """HELPER FUNCTION. Process results on train/val/test sets from model at current epoch and return as dataframe
        (containing accuracy, loss, etc.) and dictionary auc details.
        """
        sys.path.insert(0, self.args.git_dir + '/nephronetwork/2.Results/')
        from process_results import get_metrics

        df_results = pd.DataFrame()
        auc_dict = {}
        for dset in self.dsets:
            y_score, y_true, y_pred, acc, loss = self.get_dset_results(dset)
            results = get_metrics(y_score=y_score, y_true=y_true, y_pred=y_pred)

            auc_dict.update({f'{dset}_tpr': results['tpr'].tolist(), f'{dset}_fpr': results['fpr'].tolist(),
                             f'{dset}_auroc_thresholds': results['auroc_thresholds'].tolist(),
                             f'{dset}_recall': results['recall'].tolist(),
                             f'{dset}_precision': results['precision'].tolist(),
                             f'{dset}_auprc_thresholds': results['auprc_thresholds'].tolist(), "epoch": self.epoch})

            df_dset_results = pd.DataFrame({"epoch": self.epoch, "acc": acc, "loss": loss, "dset": dset,
                                            "auc": results["auc"],
                                            "auprc": results["auprc"],
                                            "tn": results["tn"],
                                            "fp": results['fp'],
                                            "fn": results['fn'],
                                            "tp": results['tp']}, index=[0])

            df_results = pd.concat([df_results, df_dset_results])

        return df_results, auc_dict

    def save_results(self):
        """For each epoch, save results, and model hyperparameters (if not already done).
        Depending on save frequency, save model (and optimizer) weights.
        """
        if not os.path.isdir(curr_results_dir):
            os.makedirs(curr_results_dir)
        df_epoch_results, auc_details = self.preprocess_results()

        # Save results
        if os.path.exists(results_summary_path):
            df_results_accum = pd.read_csv(results_summary_path)
            with open(auc_path) as f:
                auc_accum = json.load(f)
        else:
            df_results_accum = pd.DataFrame()
            auc_accum = {}

        df_results_accum = pd.concat([df_results_accum, df_epoch_results])
        df_results_accum.to_csv(results_summary_path, index=False)

        auc_accum.update(auc_details)
        with open(auc_path, 'w') as outfile:
            json.dump(auc_accum, outfile, sort_keys=True, indent=4)

    def process_results(self, epoch, dsets, y_train, y_val, y_test):
        """Main method of Results."""
        self.epoch = epoch
        self.dsets = dsets
        self.concat_results()
        self.verify_length(y_train, y_val, y_test)
        self.save_results()


# Additional helper functions (can be used if needed)
def get_id(cov):
    return cov.split("_")[0]


def get_handedness(cov):
    return cov.split("_")[4]


# Main Method
def main():
    print(timestamp)
    args = parseArgs()
    modifyArgs(args)

    # Path to data
    data_path = args.git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190617.pickle"

    # Import modules from git repository
    sys.path.insert(0, args.git_dir + '/nephronetwork/0.Preprocess/')
    sys.path.insert(0, args.git_dir + '/nephronetwork/1.Models/siamese_network/')

    from load_dataset_LE import load_dataset

    # Load data
    X_train, y_train, cov_train, X_test, y_test, cov_test = load_dataset(
        views_to_get=args.view,
        sort_by_date=True,
        pickle_file=data_path,
        contrast=args.contrast,
        split=args.split,
        get_cov=True,
        bottom_cut=args.bottom_cut,
        etiology=args.etiology,
        crop=args.crop,
        git_dir=args.git_dir
    )
    # Prepare data into sequences
    X_train, y_train, cov_train, X_test, y_test, cov_test = prepare_data_into_sequences(X_train, y_train, cov_train,
                                                                                        X_test, y_test, cov_test,
                                                                                        single_visit=args.single_visit)
    # Split into train/val/test sets.
    X_train, y_train, cov_train, val_X, y_val, val_cov = make_validation_set(X_train, y_train, cov_train)

    # Train model
    train(args, X_train, y_train, cov_train, X_test, y_test, cov_test, val_X, y_val, val_cov)


if __name__ == '__main__':
    main()
