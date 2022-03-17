import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.average_prediction import SiamNetAvgPred
from models.baseline import SiamNet
from models.conv_pooling import SiamNetConvPooling
from models.ensemble import Ensemble
from models.lstm import SiamNetLSTM
from models.tsm import SiamNetTSM
from utilities.custom_logger import FriendlyCSVLogger
from utilities.data_visualizer import plot_umap
from utilities.dataset_prep import KidneyDataModule
from utilities.kornia_augmentation import create_augmentation_str, instantiate_augmenter

warnings.filterwarnings('ignore')

SEED = 42
model_name = ""
MODEL_TYPES = ("baseline", "avg_pred", "conv_pool", "lstm", "tsm", "baseline_efficientnet", "ensemble")

project_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/"
results_dir = f"{project_dir}results/"

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
def parse_args():
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
    parser.add_argument("--json_ui_test", default="UIonly_moreDat_rootfilenames_20220110.json",
                        help="Json file of held-out, retrospective University of Iowa data")
    parser.add_argument("--json_chop_test", default="CHOP_rootfilenames_20220108.json",
                        help="Json file of held-out, retrospective Children's Hospital of Philadelphia data")

    parser.add_argument("--json_prenatal", default="Prenatal_rootfilenames_20220109.json",
                        help="Json file of SickKids prenatal ultrasounds")
    parser.add_argument("--json_postnatal", default="Postnatal_rootfilenames_Noage_20220315.json",
                        help="Json file of SickKids postnatal ultrasounds of those with prenatal scans.")

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

    parser.add_argument("--gradient_clip_norm", default=1, type=float,
                        help="Norm value for gradient clipping via vector norm.")
    parser.add_argument("--precision", default=32, type=int, help="Precision for training (32/16).")

    parser.add_argument('--augment_training', action="store_true", default=False, help="Allow image augmentation.")
    parser.add_argument('--augment_probability', default=0.5, type=float,
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


def modify_args(args):
    """Overwrite user-inputted model parameters.
    """
    global model_name

    # Model hyperparameters
    """
    args.lr = 0.0001
    args.batch_size = 16
    args.adam = True
    args.momentum = 0.9
    args.weight_decay = 0.005
    args.weighted_loss = 0.5
    args.dropout_rate = 0.5
    args.include_cov = False
    args.stop_epoch = 24
    args.output_dim = 128

    args.gradient_clip_norm = 1
    args.precision = 32
    
    # Image augmentation
    args.augment_training = True
    args.augment_probability = 0.5
    args.normalize = True
    args.random_rotation = False
    args.color_jitter = False
    args.random_gaussian_blur = True
    args.random_motion_blur = False
    args.random_noise = False
    """

    args.load_hyperparameters = False
    args.pretrained = True

    # Choose model
    args.model = MODEL_TYPES[2]

    if args.model == "baseline":
        model_name = "Siamese_Baseline"
    elif args.model == 'baseline_efficientnet':
        model_name = "Siamese_EfficientNet"
    else:
        model_name = f"Siamese_{args.model.upper()}"

    assert args.model in MODEL_TYPES

    # Data parameters
    if "baseline" in args.model:  # for single-visit models
        args.single_visit = True
        args.test_last_visit = True
    else:  # for multiple-visit models
        args.single_visit = False
        args.test_last_visit = False
        args.accum_grad_batches = args.batch_size
        args.batch_size = 1

    args.balance_classes = False
    args.num_workers = 4

    # Test set parameters
    args.test_only = True
    args.include_test = True
    args.ordered_split = False

    # Validation set parameters
    args.include_validation = not args.test_only and False
    args.cv = args.include_validation and True
    args.num_folds = 5 if (args.cv and args.include_validation) else 1

    if not args.cv and args.include_validation:
        args.ordered_validation = True  # train and validation split


def get_hyperparameters(args):
    """Returns dictionary of model hyperparameters from args. Loads previous hyperparameters if specified by args."""
    if args.accum_grad_batches is None:
        args.accum_grad_batches = 1

    hyperparams = {'lr': args.lr, "batch_size": args.batch_size * args.accum_grad_batches,
                   'adam': args.adam,
                   'momentum': args.momentum,
                   'weight_decay': args.weight_decay,
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
        load_hyperparameters(hyperparams)
    return hyperparams


def load_hyperparameters(hyperparameters: dict):
    """Load in previously found the best hyperparameters and update <hyperparameters> in-place.

    :param hyperparameters: existing dictionary of model hyperparameters
    """
    best_param_dir = None
    if hyperparameters['model'] == MODEL_TYPES[1]:
        best_param_dir = f"{results_dir}Siamese_AvgPred_grid_search(2022-02-02)"
    elif hyperparameters['model'] == MODEL_TYPES[2]:
        best_param_dir = f"{results_dir}Siamese_ConvPool_grid_search(2022-02-04)"
    elif hyperparameters['model'] == MODEL_TYPES[3]:
        best_param_dir = f"{results_dir}Siamese_LSTM_grid_search(2022-02-06)"
    elif hyperparameters['model'] == MODEL_TYPES[4]:
        best_param_dir = f"{results_dir}Siamese_TSM_grid_search(2022-02-05)"

    if best_param_dir is not None and os.path.exists(f"{best_param_dir}/best_parameters.json"):
        with open(f"{best_param_dir}/best_parameters.json", "r") as param_file:
            old_params = json.load(param_file)
        hyperparameters['stop_epoch'] = old_params['epoch'] + 1
        old_params = {k: v for k, v in list(old_params.items()) if k in hyperparameters and k != "stop_epoch"}
        hyperparameters.update(old_params)
        print("Previous hyperparameters loaded successfully!")
        print(hyperparameters)


def instantiate_model(model_type, hyperparams=None, pretrained=False, from_baseline=False):
    """Instantiate model based on arguments. Insert hyperparameters. If specified by arguments, add augmentation to
    model."""
    global curr_results_dir

    augmenter = instantiate_augmenter(hyperparams)

    model_dict = {
        MODEL_TYPES[0]: SiamNet,
        MODEL_TYPES[1]: SiamNetAvgPred,
        MODEL_TYPES[2]: SiamNetConvPooling,
        MODEL_TYPES[3]: SiamNetLSTM,
        MODEL_TYPES[4]: SiamNetTSM,
        # 'baseline_efficientnet': SiameseEfficientNet,
    }

    if model_type != "ensemble":
        model = model_dict[model_type]
    else:
        model = None

    # Load weights
    if pretrained:
        folders = {
            MODEL_TYPES[0]: f"{results_dir}/final/Siamese_Baseline_2022-02-01_23-20-51/fold0/version_0",
            MODEL_TYPES[1]: f"{results_dir}/final/Siamese_AvgPred_2022-02-07_23-44-50/fold0/version_0",
            MODEL_TYPES[2]: f"{results_dir}/final/Siamese_ConvPooling_2022-02-05_15-11-15/fold0/version_0",
            MODEL_TYPES[3]: f"{results_dir}/final/Siamese_LSTM_2022-02-09_21-53-36/fold0/version_0",
            MODEL_TYPES[4]: f"{results_dir}/final/Siamese_TSM_2022-02-21_14-49-12/fold0/version_0",
        }

        if model_type == 'ensemble':
            for _type, _model in model_dict.items():
                folder = folders[_type]
                model_dict[_type] = model_dict[_type].load_from_checkpoint(f"{folder}/checkpoints/last.ckpt",
                                                                           hparams_file=f"{folder}/hparams.yaml")
            model = Ensemble(hyperparams, augmenter, models=model_dict)
        else:
            folder = folders[model_type if not from_baseline else "baseline"]
            model = model.load_from_checkpoint(f"{folder}/checkpoints/last.ckpt",
                                               hparams_file=f"{folder}/hparams.yaml"
                                               )
            # model.load("C:/Users/Stanley Hua/SickKids/Lauren Erdman - HN_Stanley/ModelWeights/NoFinalLayerFineTuneNoCov_v2_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covFalse_OSFalse_30thEpoch.pth")
    else:
        model = model(model_hyperparams=hyperparams, augmentation=augmenter)
    return model


def update_paths(model_type, pretrained=False):
    """Update global PATH variables to reflect type of model used."""
    global curr_results_dir, training_info_path, auc_path, results_dir, results_summary_path, model_name
    if model_type == 'avg_pred':
        model_name = "Siamese_AvgPred"
    elif model_type == "conv_pool":
        model_name = "Siamese_ConvPooling"
    elif model_type == "lstm":
        model_name = "Siamese_LSTM"
    elif model_type == "tsm":
        model_name = "Siamese_TSM"
    elif model_type == "stgru":
        model_name = "Siamese_STGRU"
    elif model_type == 'baseline_efficientnet':
        model_name = 'Siamese_EfficientNet'
    else:  # baseline single-visit
        model_name = "Siamese_Baseline"

    if pretrained:
        model_name = f"Pretrained_{model_name}"

    curr_results_dir = f"{results_dir}{model_name}_{timestamp}/"
    training_info_path = f"{curr_results_dir}info.csv"
    auc_path = f"{curr_results_dir}auc.json"
    results_summary_path = f"{curr_results_dir}history.csv"


def run(hyperparams, args, dm, fold=0, checkpoint=True, tune_hyperparams=False, version_name=None,
        train=True, test=True):
    """Code to run training and/or testing."""
    print(f"Fold {fold + 1}/{args.num_folds} Starting...")
    model = instantiate_model(hyperparams['model'], hyperparams, args.pretrained)

    # Loggers
    csv_logger = FriendlyCSVLogger(f"{curr_results_dir}", name=f'fold{fold}', version=version_name)
    tensorboard_logger = TensorBoardLogger(f"{curr_results_dir}", name=f"fold{fold}", version=csv_logger.version)

    # Callbacks
    callbacks = []
    if checkpoint:
        callbacks.append(
            ModelCheckpoint(dirpath=f"{curr_results_dir}fold{fold}/version_{csv_logger.version}/checkpoints",
                            # monitor="val_loss",
                            save_last=True))
    if tune_hyperparams:
        callbacks.append(TuneReportCallback({'val_loss': 'val_loss'}, on='validation_end'))

    trainer = Trainer(default_root_dir=f"{curr_results_dir}fold{fold}/version_{csv_logger.version}",
                      gpus=1,
                      num_sanity_val_steps=0,
                      # log_every_n_steps=100,
                      accumulate_grad_batches=None if 'baseline' in hyperparams['model'] else hyperparams['batch_size'],
                      precision=hyperparams['precision'],
                      gradient_clip_val=hyperparams['gradient_clip_norm'],
                      max_epochs=hyperparams['stop_epoch'],
                      enable_checkpointing=checkpoint,
                      # stochastic_weight_avg=True,
                      callbacks=callbacks,
                      logger=[csv_logger, tensorboard_logger],
                      # fast_dev_run=True,
                      )
    if train:
        dm.fold = fold
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        trainer.fit(model, train_dataloader=train_loader,
                    val_dataloaders=val_loader if args.include_validation else None)

    if test and not args.cv:
        trainer.test(model=model,
                     test_dataloaders=[dm.test_dataloader(), dm.st_test_dataloader(), dm.stan_test_dataloader(),
                                       dm.ui_test_dataloader(), dm.chop_test_dataloader()])


def extract(dataloaders, model, model_type="baseline", which="embed", save_dir=None, dset_name=None):
    """Return dictionary or list of dictionaries, containing model prediction or high-dimensional embeddings for
    specified dataloader/s.

    :param dataloaders torch dataloader, or sequence-like of dataloaders
    :param model torch model
    :param model_type name of model
    :param which specify to extract embedding or output (probabilities)
    :param save_dir directory to save plot of embeddings
    :param dset_name plotting names for dataset/s (used in UMAP plot)
    """
    # If sequence of dataloaders, extract for each
    if isinstance(dataloaders, list):
        outputs = []
        for i, dl in enumerate(dataloaders):
            name = None if dset_name is None else dset_name[i]
            outputs.append(extract(dl, model, model_type=model_type,
                                   save_dir=save_dir, which=which, dset_name=name))
        return outputs

    if which == "embed":
        print(f"Extracting embeddings{f' for {dset_name}' if dset_name is not None else ''}...")
        out, labels, ids = model.extract_embeddings(dataloaders)
        plot_umap(out, labels, save_dir=f"{project_dir}/figures/",
                  plot_name="UMAP of " + f"{dset_name} " if dset_name is not None else "" + f"Test Set ({model_type})")
    else:  # outputs
        print(f"Performing inference{f' on {dset_name}' if dset_name is not None else ''}...")
        out, labels, ids = model.extract_preds(dataloaders)

    out_name = "embed" if which == "embed" else "pred"

    results = {out_name: out.tolist(), "label": labels.tolist(), "ids": ids.tolist()}
    return results


# Main Methods
def main(inference: bool = False, **kwargs):
    """Main method for training, or inference on test set.

    :param inference Perform training if false. Otherwise, perform inference using pretrained model and save results.
    """
    global curr_results_dir, project_dir

    # Parse arguments
    args = parse_args()
    modify_args(args)
    args.augmentations_str = create_augmentation_str(args)
    update_paths(args.model, args.pretrained)
    hyperparams = get_hyperparameters(args)

    # Set up data
    data_params = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': args.num_workers,
                   'pin_memory': True,
                   'persistent_workers': True if args.num_workers else False}
    dm = KidneyDataModule(args, data_params)
    dm.setup('fit')
    dm.setup('test')

    if not inference:
        # If folder is non-existent, create folder for storing results
        if not os.path.isdir(curr_results_dir):
            os.makedirs(curr_results_dir)

        # Perform training
        for fold in range(5 if args.cv else 1):
            run(hyperparams, args=args, dm=dm, fold=fold,
                checkpoint=not args.include_validation,
                train=not args.test_only, test=args.include_test)
    else:
        assert args.pretrained
        curr_results_dir = f"{results_dir}/final/"
        from_baseline = False

        assert 'which' in kwargs and kwargs['which'] in ['embed', 'pred']
        suffix = kwargs['suffix'] if 'suffix' in kwargs else ''

        # Get model and dataloaders
        model_type = hyperparams['model']
        model = instantiate_model(model_type, hyperparams, True, from_baseline=from_baseline)
        test_loaders = [
            # dm.test_dataloader(), dm.st_test_dataloader(), dm.stan_test_dataloader(),
            # dm.ui_test_dataloader(), dm.chop_test_dataloader(),
            dm.prenatal_test_dataloader(),
            dm.postnatal_test_dataloader()
        ]
        # test_names = ["SK Test", "SK Silent Trial", "Stanford", "Iowa", "CHOP", "Prenatal"]
        test_names = ["Prenatal", "Postnatal"]

        # Extract embedding/predictions
        result_dicts = extract(test_loaders, model, model_type, which=kwargs['which'],
                               dset_name=test_names,
                               save_dir=f"{project_dir}/figures/")

        # Store results
        # filename_map = {0: f"sk_test", 1: "st", 2: "stan", 3: "uiowa", 4: "chop", 5: "prenatal"}
        filename_map = {0: "prenatal", 1: "postnatal"}
        all_results = {}
        for i in range(len(result_dicts)):
            all_results[filename_map[i]] = result_dicts[i]

        with open(f"{curr_results_dir}/{'pretrained-' * from_baseline}{model_type}-test_output-{kwargs['which']}{suffix}.json", "w") as f:
            json.dump(all_results, f, indent=4)

    print("Finished.")


if __name__ == '__main__':
    main(inference=True,
         which="pred",
         suffix="_postnatal-prenatal")
