"""Perform Randomized Grid Search for hyperparameters.
"""
import os
import random
import shutil
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import yaml


class GridSearch:
    model_type_: str
    timestamp_: str
    grid_search_dir_: str
    metric: str
    tested_combinations: pd.DataFrame

    def __init__(self, model_type_, timestamp_, grid_search_dir_, metric='loss'):
        self.model_type = model_type_
        self.timestamp = timestamp_
        self.grid_search_dir = grid_search_dir_
        self.metric = metric
        self.tested_combinations = self.get_tested_hyperparams()

    def sample_hyperparams(self, possible_hyperparams: dict):
        """Given search space, randomly sample a set of hyperparameters to test that has not been tested.
        If lists are given, an item will be selected. Otherwise, will remain as is.
        """
        for i in range(1000):
            sampled_params = {}
            for u in possible_hyperparams.keys():
                v = possible_hyperparams[u]
                if isinstance(v, list):
                    sampled_params[u] = random.choice(v)
                else:
                    sampled_params[u] = v

            if not self.is_hyperparam_used(sampled_params):
                row = pd.DataFrame(sampled_params, index=[0])
                self.tested_combinations = pd.concat([self.tested_combinations, row], ignore_index=True)
                return sampled_params

            if i == 1000:  # upper limit, otherwise assume all have been tested
                return None

    def is_hyperparam_used(self, sample_hyperparams: dict):
        """Checks if sampled hyperparams have already been tested"""
        # Get intersecting columns
        search_columns = set(list(sample_hyperparams.keys()))
        available_columns = set(self.tested_combinations.columns.tolist())
        cols = list(search_columns.intersection(available_columns))

        return [sample_hyperparams[c] for c in cols] in gridSearch.tested_combinations[cols].values.tolist()

    def get_tested_hyperparams(self):
        dirs_ = self.get_training_dirs()

        df_accum = pd.DataFrame()
        for dir_ in dirs_:
            _hyperparams = get_hyperparams(dir_)
            row = pd.DataFrame(_hyperparams, index=[0])
            df_accum = pd.concat([df_accum, row])

        return df_accum

    def get_training_dirs(self):
        """Return list of absolute paths to all directories made during hyperparameter search and move to
        grid search directory if not already done."""
        try:  # Move existing folders to grid search directory
            timestamp_split = self.timestamp.split('-')
            all_dirs = glob(f"{project_dir}/results/*")

            if len(timestamp_split) <= 1:
                next_timestamp = "None"
            else:
                next_timestamp = f"{timestamp_split[0]}-{timestamp_split[1]}-{(int(timestamp_split[-1]) + 1)}"

            temp_folders = []
            for i in all_dirs:
                if (self.timestamp in i or next_timestamp in i) and ("grid_search" not in i) and (self.model_type in i):
                    temp_folders.append(i)

            for folder in temp_folders:
                if len(os.listdir(folder)) != 1:
                    shutil.move(folder, self.grid_search_dir)
        except FileNotFoundError:
            pass
        return [file for file in glob(f"{self.grid_search_dir}/*") if "csv" not in file and "json" not in file]

    def find_best_models(self):
        """If cross-fold validation done, average over epochs. Get row with the lowest validation loss.

        @return dataframe containing the best validation set results, where each row corresponds to a tested model 
                hyperparameters
        """
        result_directories = self.get_training_dirs()

        df_accum = pd.DataFrame()
        for dir_ in result_directories:
            df = aggregate_fold_histories(dir_)

            best_score = min(df[f"val_{self.metric}"]) if self.metric == 'loss' else max(df[f"val_{self.metric}"])
            df_best_epoch = df[df[f"val_{self.metric}"] == best_score]

            # Add in parameters
            params = get_hyperparams(dir_)
            for key in params:
                if key == "insert_where" and datetime.strptime(dir_.split('_')[-2], "%Y-%m-%d").day < 8:
                    params[key] = 0
                df_best_epoch[key] = params[key]

            df_best_epoch["dir"] = dir_
            df_accum = pd.concat([df_accum, df_best_epoch])

        print(df_accum)

        return df_accum

    def perform_grid_search(self, search_space, n=None):
        """
        Performs randomized grid search on given specified parameter lists. Tested parameter combinations are saved.
        :param search_space: dictionary, where keys are argument names for training script and values are potential
            values to search.
        :param n: number of trials to run
        """
        if n == 0:
            return

        params = self.sample_hyperparams(search_space)

        if params is None:
            return

        print(params)
        command_str = f'python "{project_dir}/op/model_training.py" '
        for u in params.keys():
            v = params[u]
            if not isinstance(v, bool):
                command_str += f"--{u} {v} "
            elif v is True:
                command_str += f"--{u} "

        subprocess.run(command_str, shell=True)

        if n is not None:
            self.perform_grid_search(search_space, n - 1)

    def save_grid_search_results(self):
        df = self.find_best_models()

        if len(df) == 0:
            return

        best_score = min(df[f"val_{self.metric}"]) if self.metric == 'loss' else max(df[f"val_{self.metric}"])
        best_model = df[df[f"val_{self.metric}"] == best_score].iloc[0]
        best_model.to_json(f"{self.grid_search_dir}/best_parameters.json")

        df = df.sort_values(by=[f"val_{self.metric}"])
        df.to_csv(f"{self.grid_search_dir}/grid_search({self.timestamp}).csv", index=False)


def aggregate_fold_histories(dir_: str):
    """Get history for each epoch and aggregate them (mean across epochs)."""
    histories = glob(f"{dir_}/*/*/history.csv")

    df = pd.DataFrame()
    for history in histories:
        fold = history.split(os.sep)[-3][-1]
        df_fold = pd.read_csv(history)
        df_fold['fold'] = fold
        df = pd.concat([df, df_fold], ignore_index=True)

    df = df.groupby(by=["epoch"]).mean().reset_index()

    for col in df.columns.tolist():
        if "_" not in col or "loss" in col:
            continue
        if ('train' not in col) and ('test' not in col) and ('val' not in col):
            continue

        if all(df[col] < 1):
            df[col] = (df[col] * 100).round(decimals=2)

    return df


def get_hyperparams(dir_: str):
    """Get hyperparameters."""
    files = list(Path(dir_).rglob("hparams.yaml"))

    with open(files[0], "r") as stream:
        hparams = yaml.load(stream, yaml.Loader)

    if "loss_weights" in hparams and isinstance(hparams["loss_weights"], tuple):
        weights = hparams.pop("loss_weights")
        hparams["weighted_loss"] = weights[-1]

    if 'augmentation_type' in hparams:
        augmentations_str = hparams.pop('augmentation_type')
        hparams['augmentations_str'] = augmentations_str

    return hparams


if __name__ == "__main__":
    # Global variables
    project_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/"
    model_type = "Siamese_TSM"
    keep_best_weights = False

    timestamp = datetime.now().strftime("%Y-%m-%d")
    grid_search_dir = f"{project_dir}/results/{model_type}{'_' if (len(model_type) > 0) else ''}grid_search({timestamp})/"

    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)

    search_space = {
        'lr': [1e-3, 1e-5, 1e-4],
        "batch_size": [1, 16, 64],
        'adam': [True, False],
        'momentum': [0.8, 0.9],
        'weight_decay': [5e-4, 5e-3],
        'weighted_loss': 0.5,
        'output_dim': [128, 256, 512],
        'dropout_rate': [0, 0.25, 0.5],
        'precision': 16,
        'augment_training': True,
        'augment_probability': 0.5,      # [0.25, 0.5, 0.75],
        'normalize': True,
        'random_rotation': [True, False],
        'color_jitter': [True, False],
        'random_gaussian_blur': True,
        'random_motion_blur': [True, False],
        'random_noise': [True, False],
        'model': ['lstm', "baseline", "avg_pred", "conv_pool", "tsm"]
    }

    gridSearch = GridSearch(model_type, timestamp, grid_search_dir, metric='auprc')
    gridSearch.perform_grid_search(search_space, n=12)
    gridSearch.save_grid_search_results()
