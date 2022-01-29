import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from models.baseline_pl import SiamNet
from models.conv_pooling import SiamNetConvPooling
from models.lstm import SiameseLSTM
from models.tsm import SiamNetTSM
from utilities.custom_logger import FriendlyCSVLogger
from utilities.dataset_prep import KidneyDataModule
from utilities.data_visualizer import plot_umap
from utilities.kornia_augmentation import DataAugmentation

warnings.filterwarnings('ignore')

SEED = 42
model_name = ""
MODEL_TYPES = ("baseline", "avg_pred", "conv_pooling", "lstm", "tsm")

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


# Defining data/model parameters
def parseArgs():
    """When running script, parse user input for argument values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--git_dir", default="C:/Users/Stanley Hua/projects/")
    parser.add_argument("--data_dir", default='C:/Users/Stanley Hua/SickKids/Lauren Erdman - HN_Stanley/',
                        help="Directory where data is located")
    parser.add_argument("--dir", default="./", help="Directory to save model checkpoints to")
    parser.add_argument("--checkpoint", default="", help="Path to load pretrained model checkpoint from")
    parser.add_argument('--pretrained', action="store_true",
                        help="Use pretrained model with cross validation if cv requested")

    parser.add_argument("--json_infile", default="HNTrain_rootfilenames_updated20220123.json",
                        help="Json file of model training data")
    parser.add_argument("--json_st_test", default="newSTonly_rootfilenames_20211229.json",
                        help="Json file of held-out, prospective silent trial data")
    parser.add_argument("--json_stan_test", default="StanfordOnly_rootfilenames_20211229.json",
                        help="Json file of held-out, retrospective Stanford data")
    parser.add_argument("--json_ui_test", default="UIonly_rootfilenames_20211229.json",
                        help="Json file of held-out, retrospective University of Iowa data")

    parser.add_argument("--model", default="baseline",
                        help="Choose model to run from the following: (baseline, conv_pool, lstm, tsm, stgru)")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.005, type=float, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument('--adam', action="store_true", help="Use Adam optimizer instead of SGD")
    parser.add_argument("--output_dim", default=256, type=int, help="output dim for last linear layer")
    parser.add_argument("--include_cov", action="store_true",
                        help="Include covariates (age at ultrasound and kidney side (R/L)) in model.")
    parser.add_argument("--accum_grad_batches", default=None, help="Number of batches to accumulate gradient.")
    parser.add_argument("--dropout_rate", default=0, type=float,
                        help="Dropout rate to apply on last linear layers during training.")
    parser.add_argument("--weighted_loss", default=0.5, type=float,
                        help="Weight to assign to positive class in calculating loss.")

    parser.add_argument("--gradient_clip_norm", default=0., type=float,
                        help="Norm value for gradient clipping via vector norm.")
    parser.add_argument("--precision", default=32, type=int, help="Precision for training (32/16).")

    parser.add_argument('--augment_training', action="store_true", default=False, help="Allow image augmentation.")
    parser.add_argument('--augment_probability', default=0., type=float,
                        help="Probability of augmentations during training.")
    parser.add_argument('--normalize', action="store_true", default=False, help="Add normalization to augmentations.")
    parser.add_argument('--random_rotation', action="store_true", default=False, help="Add rotation to augmentations.")
    parser.add_argument('--color_jitter', action="store_true", default=False, help="Add color jitter to augmentations.")
    parser.add_argument('--random_gaussian_blur', action="store_true", default=False,
                        help="Add gaussian blur to augmentations.")
    parser.add_argument('--random_motion_blur', action="store_true", default=False,
                        help="Add motion blur to augmentations.")
    parser.add_argument('--random_noise', action="store_true", default=False, help="Add random noise to augmentations.")

    parser.add_argument("--num_workers", default=1, type=int, help="Number of CPU workers")
    parser.add_argument("--split", default=0.7, type=float, help="proportion of dataset to use as training")
    parser.add_argument('--ordered_split', action="store_true", default=False,
                        help="Split data into training and testing set by time.")
    parser.add_argument('--ordered_validation', action="store_true", default=False,
                        help="Split training data into training and validation set by time.")
    parser.add_argument("--bottom_cut", default=0.0, type=float, help="proportion of dataset to cut from bottom")
    parser.add_argument('--cv', action='store_true', help="Flag to run cross validation")
    parser.add_argument("--stop_epoch", default=40, type=int,
                        help="If not running cross validation, which epoch to finish with")
    parser.add_argument("--save_frequency", default=100, type=int, help="Save model weights every <x> epochs")
    parser.add_argument("--train_only", action="store_true", help="Only fit train/val")
    parser.add_argument("--test_last_visit", action="store_true", default=True,
                        help="For single-visit baseline, only take last visit.")

    parser.add_argument("--contrast", default=1, type=int, help="Image contrast to train on")
    parser.add_argument("--crop", default=0, type=int, help="Crop setting (0=big, 1=tight)")
    parser.add_argument("--view", default="siamese", help="siamese, sag, trans")
    parser.add_argument("--etiology", default="B", help="O (obstruction), R (reflux), B (both)")

    return parser.parse_args()


def modifyArgs(args):
    """Overwrite user-inputted model parameters.
    """
    global model_name

    # Model hyperparameters
    args.lr = 0.005
    args.batch_size = 16
    args.adam = False
    args.momentum = 0.9
    args.weight_decay = 0.0005
    args.weighted_loss = 0.5
    args.dropout_rate = 0.5
    args.include_cov = True
    args.stop_epoch = 26
    args.output_dim = 256

    args.gradient_clip_norm = 1  # or None, 1
    args.precision = 32  # or 16, 32

    args.load_hyperparameters = False
    args.pretrained = False

    # Choose model
    args.model = MODEL_TYPES[0]

    if args.model == "baseline":
        model_name = "Siamese_Baseline"
    else:
        model_name = f"Siamese_{args.model.upper()}"

    assert args.model in MODEL_TYPES

    # Data parameters
    if args.model == MODEL_TYPES[0]:  # for single-visit models
        args.single_visit = True
        args.test_last_visit = True
    else:  # for multiple-visit models
        args.single_visit = False
        args.test_last_visit = False
        args.accum_grad_batches = args.batch_size
        args.batch_size = 1

    args.balance_classes = False
    args.num_workers = 4

    # Image augmentation
    args.augment_training = True
    # args.normalize = False
    # args.random_rotation = False
    # args.color_jitter = False
    args.random_gaussian_blur = True
    # args.random_motion_blur = False
    # args.random_noise = False
    #
    args.augment_probability = 0.5

    # Test set parameters
    args.test_only = False
    args.ordered_split = False

    # Validation set parameters
    args.include_validation = not args.test_only and False
    args.cv = args.include_validation and True
    args.num_folds = 5 if (args.cv and args.include_validation) else 1

    if not args.cv and args.include_validation:
        args.ordered_validation = True  # train and validation split


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


def create_augmentation_str(args):
    """Create string from augmentations enabled."""
    if not args.augment_training:
        return ""

    augmentations = []

    if args.normalize:
        augmentations.append("normalize")
    if args.random_rotation:
        augmentations.append("rotate")
    if args.color_jitter:
        augmentations.append("color_jitter")
    if args.random_gaussian_blur:
        augmentations.append("gaussian_blur")
    if args.random_motion_blur:
        augmentations.append("motion_blur")
    if args.random_noise:
        augmentations.append("gaussian_noise")

    augmentations_str = "-".join(augmentations)

    return augmentations_str


def instantiate_augmenter(args):
    """Instantiate image data augmentation object based on arguments. Return augmenter and name (given by all
    augmentation types enabled."""
    if not args.augment_training or args.model is not 'baseline':
        return None

    augmenter = DataAugmentation(normalize=args.normalize, random_rotation=args.random_rotation,
                                 color_jitter=args.color_jitter, random_gaussian_blur=args.random_gaussian_blur,
                                 random_motion_blur=args.random_motion_blur, random_noise=args.random_noise,
                                 prob=args.augment_probability)

    return augmenter


def instantiate_model(args, hyperparams):
    """Instantiate model based on arguments. Insert hyperparameters. If specified by arguments, add augmentation to
    model."""
    global curr_results_dir
    old_checkpoint = ""

    augmenter = instantiate_augmenter(args)

    if args.model == "conv_pool":
        model = SiamNetConvPooling(model_hyperparams=hyperparams)
        old_checkpoint = f"{results_dir}/Siamese_Baseline_2022-01-03_11-48-24/model-epoch_38-fold1.pth"
    elif args.model == "lstm":
        model = SiameseLSTM(hyperparams,
                            bidirectional=True,
                            hidden_dim=512,
                            n_lstm_layers=1,
                            insert_where=None)
    elif args.model == "tsm":
        model = SiamNetTSM(model_hyperparams=hyperparams)
    elif args.model == "stgru":
        # TODO: Implement this
        raise NotImplementedError("STGRU has not yet been implemented!")
    else:  # baseline single-visit
        model = SiamNet(model_hyperparams=hyperparams, augmentation=augmenter)
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
def train(config, args, dm, fold=0, checkpoint=True, tune_hyperparams=False, version_name=None):
    print(f"Fold {fold + 1}/{args.num_folds} Starting...")
    model = instantiate_model(args, config)

    # Loggers
    csv_logger = FriendlyCSVLogger(f"{curr_results_dir}", name=f'fold{fold}', version=version_name)
    tensorboard_logger = TensorBoardLogger(f"{curr_results_dir}", name=f"fold{fold}", version=csv_logger.version)

    # Callbacks
    callbacks = []
    if checkpoint:
        callbacks.append(
            ModelCheckpoint(dirpath=f"{curr_results_dir}fold{fold}/version_{csv_logger.version}/checkpoints",
                            # monitor="val_loss",
                            save_last=True
                            ))
    if tune_hyperparams:
        callbacks.append(TuneReportCallback({'val_loss': 'val_loss'}, on='validation_end'))

    # Get dataloaders
    dm.fold = fold
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    trainer = Trainer(default_root_dir=f"{curr_results_dir}fold{fold}/version_{csv_logger.version}",
                      gpus=1, num_sanity_val_steps=0,
                      # log_every_n_steps=100,
                      accumulate_grad_batches=args.accum_grad_batches,
                      precision=args.precision,
                      gradient_clip_val=args.gradient_clip_norm,  # TODO: Test effect of precision and grad clip val
                      max_epochs=args.stop_epoch,
                      enable_checkpointing=checkpoint,
                      # stochastic_weight_avg=True,
                      callbacks=callbacks,
                      logger=[csv_logger, tensorboard_logger],
                      # fast_dev_run=True,
                      )
    folder = "C:\\Users\\Stanley Hua\\projects\\temporal_hydronephrosis\\results\\Siamese_Baseline_2022-01-27_19-59-33\\fold0\\version_0"
    model = model.load_from_checkpoint(f"{folder}/checkpoints/last.ckpt", hparams_file=f"{folder}/hparams.yaml")
    # trainer.fit(model, train_dataloader=train_loader,
    #             val_dataloaders=val_loader if args.include_validation else None)

    trainer.test(test_dataloaders=[
        dm.test_dataloader(),
        dm.st_test_dataloader(),
        dm.stan_test_dataloader(),
        dm.ui_test_dataloader()
    ],
        model=model)


def extract_embeddings(checkpoint_path, args, hyperparams, dm):
    model = instantiate_model(args, hyperparams)
    model = model.load_from_checkpoint(checkpoint_path)
    val_loader = dm.val_dataloader()
    embeds, labels, ids = model.extract_embeddings(val_loader)

    reducer = umap.UMAP(random_state=42)
    umap_embeds = reducer.fit_transform(embeds)

    plot_umap(umap_embeds, labels)


# Main Methods
def main():
    print(timestamp)
    args = parseArgs()
    modifyArgs(args)
    args.augmentations_str = create_augmentation_str(args)

    # Paths
    update_paths(args.model)
    # try:
    # If folder is non-existent, create folder for storing results
    if not os.path.isdir(curr_results_dir):
        os.makedirs(curr_results_dir)

    torch.backends.cudnn.benchmark = True

    # Save/load in the best hyperparameters if available
    args.tune_hyperparameters = False
    if not args.tune_hyperparameters:
        hyperparams = {'lr': args.lr, "batch_size": args.batch_size,
                       'adam': args.adam,
                       'momentum': args.momentum,
                       'weight_decay': args.weight_decay,
                       # 'patience': args.early_stopping_patience,
                       # 'num_epochs': args.epochs, 'stop_epoch': args.stop_epoch,
                       # 'balance_classes': args.balance_classes,
                       'include_cov': args.include_cov,
                       'output_dim': args.output_dim,
                       'dropout_rate': args.dropout_rate,
                       'weighted_loss': args.weighted_loss,
                       'stop_epoch': args.stop_epoch,
                       'gradient_clip_norm': args.gradient_clip_norm,
                       'precision': args.precision,
                       'model': args.model,
                       'augmented': args.augment_training,
                       'augment_probability': args.augment_probability,
                       'augmentations_str': args.augmentations_str
                       }

        if args.load_hyperparameters:
            load_hyperparameters(hyperparams, best_hyperparameters_folder)
            # args.save_frequency = hyperparams['stop_epoch']
            # args.stop_epoch = hyperparams['stop_epoch']

        data_params = {'batch_size': hyperparams['batch_size'],
                       'shuffle': True,
                       'num_workers': args.num_workers,
                       'pin_memory': True,
                       'persistent_workers': True if args.num_workers else False, }

        dm = KidneyDataModule(args, data_params)
        dm.setup('fit')
        dm.setup('test')

        for fold in range(5 if args.cv else 1):
            train(config=hyperparams, args=args, dm=dm, fold=fold, checkpoint=True)
    else:
        hyperparams = {'lr': tune.loguniform(1e-4, 1e-1), "batch_size": 128,
                       'adam': True,
                       'momentum': 0.9,
                       'weight_decay': 5e-4,
                       # 'patience': args.early_stopping_patience,
                       # 'num_epochs': args.epochs, 'stop_epoch': args.stop_epoch,
                       'loss_weights': (0.13, 0.87),
                       'include_cov': True,
                       'output_dim': 128,
                       'dropout_rate': 0.5}

        # scheduler = ASHAScheduler(max_t=100, grace_period=1, reduction_factor=2)
        scheduler = PopulationBasedTraining(perturbation_interval=4,
                                            hyperparam_mutations={'lr': tune.loguniform(1e-4, 1e-1),
                                                                  "batch_size": [1, 64, 128],
                                                                  'adam': [True, False],
                                                                  'momentum': [0.8, 0.9],
                                                                  'weight_decay': [5e-4, 5e-3],
                                                                  'loss_weights': [(0.5, 0.5), (0.13, 0.87)],
                                                                  'output_dim': [128, 256, 512],
                                                                  'dropout_rate': [0, 0.25, 0.5]
                                                                  })
        reporter = CLIReporter(parameter_columns=list(hyperparams.keys()), metric_columns=['val_loss'])

        train_fn_with_parameters = tune.with_parameters(train, args=args)

        analysis = tune.run(train_fn_with_parameters, resources_per_trial={'cpu': args.num_workers, 'gpu': 1},
                            metric="val_loss", mode='min', config=hyperparams, num_samples=10,
                            progress_reporter=reporter, scheduler=scheduler,
                            name='tune_baseline_pbt')

        print("Best hyperparameters found were:", analysis.best_config)


if __name__ == '__main__':
    main()
    # pass
