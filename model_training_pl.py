import argparse
import json
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from models.baseline_pl import SiamNet
from models.conv_pooling import SiamNetConvPooling
from models.lstm import SiameseLSTM
from utilities.dataset_prep import KidneyDataModule

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
    args.lr = 0.00001
    args.batch_size = 128
    args.adam = False
    args.momentum = 0.8
    args.weight_decay = 0.0005

    args.early_stopping_patience = 100
    args.save_frequency = 1000  # Save weights every x epochs
    args.include_cov = False
    args.load_hyperparameters = False
    args.pretrained = False

    # Choose model
    model_types = ["baseline", "conv_pool", "lstm", "tsm", "stgru"]
    args.model = model_types[0]

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


def choose_model(args, hyperparams):
    global curr_results_dir, best_hyperparameters_folder, device
    old_checkpoint = ""

    if args.model == "conv_pool":
        model = SiamNetConvPooling(output_dim=256, device=device, cov_layers=args.include_cov)
        old_checkpoint = f"{results_dir}/Siamese_Baseline_2022-01-03_11-48-24/model-epoch_38-fold1.pth"
    elif args.model == "lstm":
        model = SiameseLSTM(output_dim=128, batch_size=args.batch_size,
                            bidirectional=True,
                            hidden_dim=512,
                            n_lstm_layers=1,
                            insert_where=0,
                            device=device, cov_layers=args.include_cov)
    elif args.model == "tsm":
        # TODO: Implement this
        raise NotImplementedError("TSM has not yet been implemented!")
    elif args.model == "stgru":
        # TODO: Implement this
        raise NotImplementedError("STGRU has not yet been implemented!")
    else:  # baseline single-visit
        model = SiamNet(output_dim=256, cov_layers=args.include_cov, args=args, hyperparameters=hyperparams)
        old_checkpoint = f"{results_dir}/Siamese_Baseline_2022-01-03_11-48-24/model-epoch_38-fold1.pth"

    # Load weights
    if args.pretrained:
        model.load(old_checkpoint)
    return model


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
def train(model, dm, fold=0):
    global best_hyperparameters_folder

    dm.fold = fold
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    csv_logger = CSVLogger(f"{curr_results_dir}", name=f"fold{fold}", version=0)
    tensorboard_logger = TensorBoardLogger(f"{curr_results_dir}", name=f"fold{fold}", version=0)

    trainer = Trainer(default_root_dir=curr_results_dir, gpus=1,
                      accumulate_grad_batches=None,
                      precision=16,
                      gradient_clip_val=0.5,
                      num_sanity_val_steps=1,
                      max_epochs=100,
                      logger=[csv_logger, tensorboard_logger])

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(test_dataloaders=test_loader)


# Main Method
def main():
    print(timestamp)
    args = parseArgs()
    modifyArgs(args)

    # Paths
    update_paths(args.model)
    i = 1
    # try:
    # If folder is non-existent, create folder for storing results
    if not os.path.isdir(curr_results_dir):
        os.makedirs(curr_results_dir)

    torch.backends.cudnn.benchmark = True

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
    params = {'batch_size': hyperparams["batch_size"],
              'shuffle': True,
              'num_workers': args.num_workers,
              'pin_memory': True,
              'persistent_workers': True,
              }

    dm = KidneyDataModule(args, params)
    dm.prepare_data()
    dm.setup()
    model = choose_model(args, hyperparams)

    try:
        # Train model
        for fold in range(5 if args.cv else 1):  # only iterates once if not cross-fold validation
            print(f"Fold {i}/{args.num_folds} Starting...")

            train(model, dm, fold=fold)

        # if not args.test_only:
        #     plot_loss(curr_results_dir)

    except Exception as e:
        # If exception occurs, print stack trace and remove results directory.
        print(e)
        shutil.rmtree(curr_results_dir)


if __name__ == '__main__':
    main()
