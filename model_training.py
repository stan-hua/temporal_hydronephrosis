from datetime import datetime
import os
import sys
import argparse
import json
import warnings
import shutil

from sklearn.utils import shuffle

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Variable

from models.baselineSiamese import SiamNet
from models.convPooling import SiamNetConvPooling
from models.lstm import SiameseLSTM
from utilities.dataset_prep import KidneyDataset, pad_collate, prepare_data_into_sequences, make_validation_set
from utilities.results import Results
from utilities.data_visualizer import plot_loss

warnings.filterwarnings('ignore')

SEED = 42
# model_name = "Siamese_Baseline"
model_name = "Siamese_Baseline_ensemble"
# model_name = "Siamese_ConvPooling"
# model_name = "Siamese_ConvPooling_pretrained"
# model_name = "Siamese_LSTM"
best_hyperparameters_folder = None

project_dir = "C:\\Users\\Stanley Hua\\projects\\temporal_hydronephrosis\\"
results_dir = f"{project_dir}results\\"

# Paths to save results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
curr_results_dir = f"{results_dir}{model_name}_{timestamp}/"
training_info_path = f"{curr_results_dir}info.csv"
auc_path = f"{curr_results_dir}auc.json"
results_summary_path = f"{curr_results_dir}history.csv"

# Set the random seed manually for reproducibility. Set other torch-related variables.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax = torch.nn.Softmax(dim=1)

# Clear unused data
if torch.cuda.is_available():
    torch.cuda.empty_cache()


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
    # Model hyperparameters
    args.lr = 0.00001
    args.batch_size = 1
    args.early_stopping_patience = 20
    args.save_frequency = 100  # Save weights every x epochs

    # Data parameters
    args.standardize_batch_size = False
    args.single_visit = False
    args.single_target = True  # if true, only one label for each example sequence

    args.balance_classes = False

    # Test set parameters
    args.test_only = True  # if true, only perform test

    # Validation set parameters
    args.include_validation = not args.test_only and True
    args.cv = args.include_validation and True
    args.num_folds = 5 if args.cv and args.include_validation else 1

    # Choose model
    args.pretrained = True

    args.baseline = True
    args.conv = False
    args.lstm = False
    args.vgg_bn = False

    args.stgru = False
    args.tsm = False


def load_hyperparameters(hyperparameters: dict, path: str):
    """Load in previously found best hyperparameters and update <hyperparameters>.

    :param hyperparameters: existing dictionary of model hyperparameters
    :param path: path to grid search directory containing old hyperparameters stored in json
    """
    if path is not None and os.path.exists(f"{path}/best_parameters.json"):
        with open(f"{path}/best_parameters.json", "r") as param_file:
            old_params = json.load(param_file)
        old_params = {k: v for k, v in list(old_params.items()) if k in hyperparameters}
        hyperparameters.update(old_params)
        print("Previous hyperparameters loaded successfully!")


def choose_model(args):
    global model_name, best_hyperparameters_folder

    old_checkpoint = f"{project_dir}/weights/siam_checkpoint_18.pth"

    if args.baseline:
        model_name = "Siamese_Baseline"

        model = SiamNet(output_dim=256)
        if args.pretrained:
            model.load(old_checkpoint)
        return model.to(device)
    elif args.conv:
        model_name = "Siamese_ConvPooling"
        best_hyperparameters_folder = f"{results_dir}/ConvPooling_grid_search(2021-12-02)"

        model = SiamNetConvPooling(output_dim=256)
        if args.pretrained:
            model.load(old_checkpoint)
        return model.to(device)
    elif args.lstm:
        model = SiameseLSTM(output_dim=256, batch_size=args.batch_size,
                            bidirectional=True,
                            n_lstm_layers=1)
        return model.to(device)

    # Legacy Code
    sys.path.insert(0, args.git_dir + '/nephronetwork/1.Models/siamese_network/')
    from VGGResNetSiameseLSTM import SiameseCNNLstm
    if args.densenet:
        print("importing SiameseCNNLstm densenet")
        return SiameseCNNLstm("densenet").to(device)
    elif args.resnet18:
        print("importing SiameseCNNLstm resnet18")
        return SiameseCNNLstm("resnet18").to(device)
    elif args.resnet50:
        print("importing SiameseCNNLstm resnet50")
        return SiameseCNNLstm("resnet50").to(device)
    elif args.vgg:
        print("importing SiameseCNNLstm vgg")
        return SiameseCNNLstm("vgg").to(device)
    elif args.vgg_bn:
        print("importing SiameseCNNLstm vgg_bn")
        return SiameseCNNLstm("vgg_bn").to(device)
    elif args.customnet:
        print("importing SiameseCNNLstm customNet")
        return SiameseCNNLstm("custom").to(device)


# Training
def train(args, X_train, y_train, cov_train, X_test, y_test, cov_test, X_val=None, y_val=(), cov_val=(), fold=0):
    global best_hyperparameters_folder

    # Create model and hyperparameters. Load in best hyperparameters if available
    net = choose_model(args)
    hyperparams = {'lr': args.lr, "batch_size": args.batch_size,
                   'adam': args.adam,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay, 'train/test_split': args.split,
                   'patience': args.early_stopping_patience,
                   'num_epochs': args.epochs, 'stop_epoch': args.stop_epoch
                   }
    # load_hyperparameters(hyperparams, best_hyperparameters_folder)

    if args.adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                     weight_decay=hyperparams['weight_decay'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                    weight_decay=hyperparams['weight_decay'])
    params = {'batch_size': hyperparams["batch_size"],
              'shuffle': True,
              'num_workers': args.num_workers}

    # Validation/test set contains batch size of 1 to avoid interference from zero-padding varying sequence length
    val_test_params = params.copy()
    val_test_params["batch_size"] = 1

    # Be careful to not shuffle order of image seq within a patient
    X_train, y_train, cov_train = shuffle(X_train, y_train, cov_train, random_state=SEED)

    # Datasets
    training_set = KidneyDataset(X_train, y_train, cov_train)
    val_set = KidneyDataset(X_val, y_val, cov_val) if (X_val is not None) else None
    test_set = KidneyDataset(X_test, y_test, cov_test)

    # Weighted sampling
    if args.balance_classes:
        samples_weight = torch.from_numpy(training_set.get_class_proportions())
        print(samples_weight)
        sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), len(training_set))
        params["shuffle"] = False
        params["sampler"] = sampler

    # Data Loaders
    training_generator = DataLoader(training_set,
                                    collate_fn=pad_collate if args.standardize_batch_size else None,
                                    **params)
    val_generator = DataLoader(val_set,
                               collate_fn=pad_collate if args.standardize_batch_size else None,
                               **val_test_params) if (X_val is not None) else None

    test_generator = DataLoader(test_set,
                                collate_fn=pad_collate if args.standardize_batch_size else None,
                                **val_test_params)

    print("Dataset generated")

    # Results accumulator
    res = Results(y_train, y_val, y_test, args, fold=fold, results_summary_path=results_summary_path, auc_path=auc_path)

    # Variables for early stopping
    best_val_loss = 100000
    num_epochs_stagnant = 0

    for epoch in range(1, args.stop_epoch + 1):
        # Reset results accumulator
        res.reset()

        # Training & Validation
        if not args.test_only:
            print(f"Epoch {epoch}/{args.stop_epoch}")
            # Training
            net.train()
            for batch_idx, (data, target, cov) in enumerate(training_generator):
                optimizer.zero_grad()
                output = net(data.to(device))

                target = torch.tensor(target)
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
            if X_val is not None:
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
            res.process_results(epoch, ["train", "val"] if X_val is not None else ["train"], y_train, y_val, y_test)

        # Save model and optimizer weights
        if epoch % args.save_frequency == 0:
            model_path = f"{curr_results_dir}/model-epoch_{epoch}-fold{fold}.pth"
            # optimizer_path = f"{curr_results_dir}/optimizer-epoch_{epoch}.pth"

            torch.save(net.state_dict(), model_path)
            # torch.save(optimizer.state_dict(), optimizer_path)

        # Save hyperparameters if not already saved
        if not os.path.exists(training_info_path):
            df_info = pd.DataFrame(hyperparams, index=[0])
            df_info.to_csv(training_info_path, index=False)

        # Testing Set
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

        # Break if only testing
        if args.test_only:
            break

        # Print current AUPRC
        print(f"\tTrain AUPRC: {res.train_val_auprc[0]},  Val AUPRC: {res.train_val_auprc[1]}")

        # Early stopping
        if X_val is not None:
            if res.curr_val_loss <= best_val_loss:
                best_val_loss = res.curr_val_loss
                num_epochs_stagnant = 0
            else:
                num_epochs_stagnant += 1

            if num_epochs_stagnant == args.early_stopping_patience:
                print(f"Early Stopping at Epoch {epoch}...")

                # Update hyperparameters
                hyperparams["num_epochs"] = epoch
                df_info = pd.DataFrame(hyperparams, index=[0])
                df_info.to_csv(training_info_path, index=False)
                break


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
                                                                                        single_visit=args.single_visit,
                                                                                        single_target=args.single_target,
                                                                                        fix_seq_length=args.standardize_batch_size
                                                                                        )
    # Split into train-validation sets
    if args.include_validation:
        train_val_generator = make_validation_set(X_train, y_train, cov_train,
                                                  cv=args.cv, num_folds=args.num_folds)
    else:
        train_val_generator = [(X_train, y_train, cov_train, None, (), ())]

    i = 1
    try:
        # Create directory for storing results
        if not os.path.isdir(curr_results_dir):
            os.makedirs(curr_results_dir)

        # Train model
        for train_val_fold in train_val_generator:  # only iterates once if not cross-fold validation
            print(f"Fold {i}/{args.num_folds} Starting...")
            X_train, y_train, cov_train, X_val, y_val, cov_val = train_val_fold

            train(args, X_train, y_train, cov_train, X_test, y_test, cov_test, X_val, y_val, cov_val, fold=i)
            i += 1

            if args.test_only:
                break

        if not args.test_only:
            plot_loss(curr_results_dir)
    except:
        shutil.rmtree(curr_results_dir)


if __name__ == '__main__':
    main()
