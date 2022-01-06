import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

from models.baseline import SiamNet
from models.conv_pooling import SiamNetConvPooling
from models.lstm import SiameseLSTM
from utilities.data_visualizer import plot_loss
from utilities.dataset_prep import prepare_data_into_sequences, make_validation_set, create_data_loaders, parse_cov, \
    remove_unnecessary_cov, recreate_train_test_split, remove_invalid_samples
from utilities.results import Results

warnings.filterwarnings('ignore')

SEED = 42
model_name = ""

project_dir = "C:\\Users\\Stanley Hua\\projects\\temporal_hydronephrosis\\"
results_dir = f"{project_dir}results\\"

best_hyperparameters_folder = f"{results_dir}LSTM_grid_search(2021-12-11)"
best_hyperparameters_folder = f'{results_dir}Siamese_Baseline_grid_search(2022-01-02)'

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
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()


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

    # Unused arguments
    # parser.add_argument('--vgg', action='store_true', help="Run VGG16 architecture, not using this flag runs ResNet")
    # parser.add_argument('--vgg_bn', action='store_true', help="Run VGG16 batch norm architecture")
    # parser.add_argument('--densenet', action='store_true', help="Run DenseNet")
    # parser.add_argument('--resnet18', action='store_true',
    #                     help="Run ResNet18 architecture, not using this flag runs ResNet16")
    # parser.add_argument('--resnet50', action='store_true',
    #                     help="Run ResNet50 architecture, not using this flag runs ResNet50")
    parser.add_argument('--pretrained', action="store_true",
                        help="Use pretrained model with cross validation if cv requested")

    # Newly added arguments
    parser.add_argument("--model", default="baseline",
                        help="Choose model to run from the following: (baseline, conv_pool, lstm, tsm, stgru)")
    parser.add_argument("--early_stopping_patience", default=100, type=int,
                        help="If validation set specified, determine patience (num. of epochs) to allow stagnant "
                             "validation loss before stopping.")
    parser.add_argument("--save_frequency", default=100, type=int, help="Save model weights every <x> epochs")
    parser.add_argument("--include_cov", action="store_true", help="Include covariates in model.")
    return parser.parse_args()


def modifyArgs(args):
    """Overwrite user-inputted model parameters.
    """
    global model_name

    # Model hyperparameters
    args.lr = 0.0001
    args.batch_size = 13
    args.adam = True
    args.momentum = 0.8
    args.weight_decay = 0.0005

    args.early_stopping_patience = 100
    args.save_frequency = 1000  # Save weights every x epochs
    args.include_cov = False
    args.load_hyperparameters = False
    args.pretrained = False

    # Choose model
    model_types = ["baseline", "conv_pool", "lstm", "tsm", "stgru"]
    args.model = model_types[1]

    if args.model == "baseline":
        model_name = "Siamese_Baseline"
    elif args.model == "conv_pool":
        model_name = "Siamese_ConvPooling"
    else:
        model_name = f"Siamese_{args.model.upper()}"

    assert args.model in model_types

    # Data parameters
    if args.model == model_types[0]:  # for single-visit models
        args.standardize_seq_length, args.single_visit, args.single_target = False, True, False
    elif args.model == model_types[1]:
        args.standardize_seq_length, args.single_visit, args.single_target = True, False, True
        # assert args.batch_size == 1
    else:  # for multiple-visit models
        args.standardize_seq_length, args.single_visit, args.single_target = True, False, True

    args.balance_classes = False
    args.num_workers = 4

    # Test set parameters
    args.test_only = False  # if true, only perform test

    # Validation set parameters
    args.include_validation = not args.test_only and True
    args.cv = args.include_validation and True
    args.num_folds = 5 if (args.cv and args.include_validation) else 1


def load_hyperparameters(hyperparameters: dict, path: str):
    """Load in previously found the best hyperparameters and update <hyperparameters> in-place.

    @param hyperparameters: existing dictionary of model hyperparameters
    @param path: path to grid search directory containing old hyperparameters stored in json
    """
    if path is not None and os.path.exists(f"{path}/best_parameters.json"):
        with open(f"{path}/best_parameters.json", "r") as param_file:
            old_params = json.load(param_file)
        hyperparameters['stop_epoch'] = old_params['epoch']
        old_params = {k: v for k, v in list(old_params.items()) if k in hyperparameters and k != "stop_epoch"}
        hyperparameters.update(old_params)
        print("Previous hyperparameters loaded successfully!")
        print(hyperparameters)


def choose_model(args):
    global curr_results_dir, best_hyperparameters_folder, device
    old_checkpoint = ""

    if args.model == "conv_pool":
        model = SiamNetConvPooling(output_dim=256, device=device, cov_layers=args.include_cov)
        old_checkpoint = f"{results_dir}/Siamese_Baseline_2022-01-03_11-48-24/model-epoch_38-fold1.pth"
    elif args.model == "lstm":
        model = SiameseLSTM(output_dim=256, batch_size=args.batch_size,
                            bidirectional=True,
                            hidden_dim=128,
                            n_lstm_layers=1,
                            device=device, cov_layers=args.include_cov)
    elif args.model == "tsm":
        # TODO: Implement this
        raise NotImplementedError("TSM has not yet been implemented!")
    elif args.model == "stgru":
        # TODO: Implement this
        raise NotImplementedError("STGRU has not yet been implemented!")
    else:  # baseline single-visit
        model = SiamNet(output_dim=256, device=device, cov_layers=args.include_cov)
        old_checkpoint = f"{results_dir}/Siamese_Baseline_2022-01-03_11-48-24/model-epoch_38-fold1.pth"

    # Load weights
    if args.pretrained:
        model.load(old_checkpoint)
    return model.to(device)


def init_weights(m):
    """Perform Kaiming (zero-mean) initialization for conv and linear layers."""
    if not isinstance(m, torch.nn.Linear):
        return

    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    m.bias.data.fill_(0.01)


def update_paths(model_type):
    global curr_results_dir, training_info_path, auc_path, results_dir, results_summary_path, model_name
    if model_type == "conv_pool":
        model_name = "Siamese_ConvPooling"
    elif model_type == "lstm":
        model_name = "Siamese_LSTM"
    elif model_type == "tsm":
        model_name = "Siamese_TSM"
    elif model_type == "stgru":
        model_name = "Siamese_STGRU"
    else:  # baseline single-visit
        model_name = "Siamese_Baseline"

    curr_results_dir = f"{results_dir}{model_name}_{timestamp}/"
    training_info_path = f"{curr_results_dir}info.csv"
    auc_path = f"{curr_results_dir}auc.json"
    results_summary_path = f"{curr_results_dir}history.csv"


# Training
def train(args, X_train, y_train, cov_train, X_test, y_test, cov_test, X_val=None, y_val=(), cov_val=(), fold=0):
    global best_hyperparameters_folder

    # Create model. Initialize weights
    net = choose_model(args)
    net.zero_grad()
    # net.apply(init_weights)

    # Save/load in the best hyperparameters if available
    hyperparams = {'lr': args.lr, "batch_size": args.batch_size,
                   'adam': args.adam,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay, 'train/test_split': args.split,
                   'patience': args.early_stopping_patience,
                   'num_epochs': args.epochs, 'stop_epoch': args.stop_epoch,
                   'balance_classes': args.balance_classes,
                   'include_cov': args.include_cov
                   }
    if args.load_hyperparameters:
        load_hyperparameters(hyperparams, best_hyperparameters_folder)
        # args.save_frequency = hyperparams['stop_epoch']
        # args.stop_epoch = hyperparams['stop_epoch']

    if args.adam:
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['lr'],
                                     weight_decay=hyperparams['weight_decay'])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=hyperparams['lr'], momentum=hyperparams['momentum'],
                                    weight_decay=hyperparams['weight_decay'])
    params = {'batch_size': hyperparams["batch_size"],
              'shuffle': True,
              'num_workers': args.num_workers,
              'pin_memory': True,
              'persistent_workers': True,
              }

    # Validation/test set contains batch size of 1 to avoid interference from zero-padding varying sequence length
    val_test_params = params.copy()
    # val_test_params["batch_size"] = 1     # TODO: Remove this

    # Be careful to not shuffle order of image seq within a patient
    X_train, y_train, cov_train = shuffle(X_train, y_train, cov_train, random_state=SEED)
    # Datasets
    training_generator, val_generator, test_generator = create_data_loaders(X_train, y_train, cov_train, X_val, y_val,
                                                                            cov_val, X_test, y_test, cov_test,
                                                                            args, params, val_test_params)
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
                optimizer.zero_grad(set_to_none=True)

                data_dict = {}
                if args.standardize_seq_length:
                    data_dict['img'] = data[0].to(device, non_blocking=True)
                    data_dict['length'] = torch.from_numpy(data[1])
                else:
                    data_dict['img'] = data.to(device, non_blocking=True)
                if args.include_cov:
                    data_dict['cov'] = cov

                output = net(data_dict)
                target = torch.tensor(target, requires_grad=False).type(torch.LongTensor).to(device, non_blocking=True)
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
                # res.all_patient_ID_train.append(cov)

            # Validation Set
            if X_val is not None:
                net.eval()
                with torch.no_grad():
                    for batch_idx, (data, target, cov) in enumerate(val_generator):
                        net.zero_grad()
                        optimizer.zero_grad(set_to_none=True)

                        data_dict = {}
                        if args.standardize_seq_length:
                            data_dict['img'] = data[0].to(device, non_blocking=True)
                            data_dict['length'] = torch.from_numpy(data[1])
                        else:
                            data_dict['img'] = data.to(device, non_blocking=True)
                        if args.include_cov:
                            data_dict['cov'] = cov

                        output = net(data_dict)
                        target = torch.tensor(target).type(torch.LongTensor).to(device, non_blocking=True)
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
                        # res.all_patient_ID_val.append(cov)

            # Save results every epoch
            res.process_results(epoch,
                                ["train", "val"] if X_val is not None else ["train"],
                                y_train, y_val, y_test)

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
                optimizer.zero_grad(set_to_none=True)

                data_dict = {}
                if args.standardize_seq_length:
                    data_dict['img'] = data[0].to(device, non_blocking=True)
                    data_dict['length'] = torch.from_numpy(data[1])
                else:
                    data_dict['img'] = data.to(device, non_blocking=True)
                if args.include_cov:
                    data_dict['cov'] = cov

                output = net(data_dict)
                target = torch.tensor(target).type(torch.LongTensor).to(device)
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

    # Paths
    update_paths(args.model)
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

    # Parse for covariates
    if isinstance(cov_train, list) and isinstance(cov_test, list):
        func_parse_cov = partial(parse_cov, age=True, side=True, sex=False)
        cov_train = list(map(func_parse_cov, cov_train))
        cov_test = list(map(func_parse_cov, cov_test))

    # Remove samples without proper covariates
    X_train, y_train, cov_train = remove_invalid_samples(X_train, y_train, cov_train)
    X_test, y_test, cov_test = remove_invalid_samples(X_test, y_test, cov_test)

    # Recreate data split
    X_train, y_train, cov_train, X_test, y_test, cov_test = recreate_train_test_split(X_train, y_train, cov_train,
                                                                                      X_test, y_test, cov_test)

    # Prepare data into sequences
    X_train, y_train, cov_train, X_test, y_test, cov_test = prepare_data_into_sequences(X_train, y_train, cov_train,
                                                                                        X_test, y_test, cov_test,
                                                                                        single_visit=args.single_visit,
                                                                                        single_target=args.single_target)
    # Remove ID and imaging date from covariates
    remove_unnecessary_cov(cov_train)
    remove_unnecessary_cov(cov_test)

    # Split into train-validation sets
    if args.include_validation:
        train_val_generator = make_validation_set(X_train, y_train, cov_train,
                                                  cv=args.cv, num_folds=args.num_folds)
    else:
        train_val_generator = [(X_train, y_train, cov_train, None, (), ())]

    i = 1
    # try:
    # If folder is non-existent, create folder for storing results
    if not os.path.isdir(curr_results_dir):
        os.makedirs(curr_results_dir)

    torch.backends.cudnn.benchmark = True

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
    # except Exception as e:
    #     # If exception occurs, print stack trace and remove results directory.
    #     print(e)
    #     shutil.rmtree(curr_results_dir)


if __name__ == '__main__':
    main()
