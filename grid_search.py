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
    timestamp_: str
    grid_search_dir_: str
    model_type_: str
    randomized: bool

    def __init__(self, model_type_, timestamp_, grid_search_dir_):
        self.model_type = model_type_
        self.timestamp = timestamp_
        self.grid_search_dir = grid_search_dir_
        self.metric = 'loss'
        self.tested_combinations = self.get_tested_hyperparams()

    def sample_hyperparams(self, possible_hyperparams: dict):
        """Given search space, randomly sample a set of hyperparameters to test that has not been tested.
        If lists are given, an item will be selected. Otherwise, will remain as is.
        """
        used = True

        sampled_params = None

        i = 0
        while used:
            sampled_params = {}
            for u in possible_hyperparams.keys():
                v = possible_hyperparams[u]
                if isinstance(v, list):
                    sampled_params[u] = random.choice(v)
                else:
                    sampled_params[u] = v

            if not self.is_hyperparam_used(sampled_params):
                used = False

            if i == 1000:  # upper limit, otherwise assume all have been tested
                return None

        return sampled_params

    def is_hyperparam_used(self, sample_hyperparams: dict):
        """Checks if sampled hyperparams have already been tested"""
        all_compared = pd.DataFrame(sample_hyperparams, index=[0]).isin(self.tested_combinations.values.ravel())
        row_comparisons = all_compared.mean(axis=1)
        return any(row_comparisons == 1)

    def get_tested_hyperparams(self):
        dirs_ = self.get_training_dirs()

        df_accum = pd.DataFrame()
        for dir_ in dirs_:
            hyperparams = get_hyperparams(dir_)
            row = pd.DataFrame(hyperparams, index=[0])
            df_accum = pd.concat([df_accum, row])

        return df_accum

    def get_training_dirs(self):
        """Return list of absolute paths to all directories made during hyperparameter search and move to
        grid search directory if not already done."""
        try:  # Move existing folders to grid search directory
            timestamp_split = self.timestamp.split('-')
            all_dirs = glob(f"{project_dir}/results/*")
            next_timestamp = f"{timestamp_split[0]}-{timestamp_split[1]}-{(int(timestamp_split[-1]) + 1)}"
            temp_folders = []
            for i in all_dirs:
                if (self.timestamp in i or next_timestamp in i) and ("grid_search" not in i) and (self.model_type in i):
                    temp_folders.append(i)

            for folder in temp_folders:
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
            df_best_epoch = df[df[f"val_{self.metric}"] == min(df[f"val_{self.metric}"])]

            # Add in parameters
            params = get_hyperparams(dir_)
            for key in params:
                df_best_epoch[key] = params[key]

            df_best_epoch["dir"] = dir_
            df_accum = pd.concat([df_accum, df_best_epoch])

        print(df_accum)

        return df_accum

    def perform_grid_search(self, search_space, n=None):
        """
        Performs randomized grid search on given specified parameter lists. Tested parameter combinations are saved.
        @param lrs_: learning rates to test
        @param batch_sizes_: batch sizes to test
        @param momentums_: SGD momentums to test (if adam is True)
        @param adams_: If contains True, will run adam. If contains False, will run SGD.
        @param num_epochs_: number of epochs
        @param n: maximum number of parameter combinations to test
        """
        params = self.sample_hyperparams(search_space)
        print(params)
        command_str = f'python "{project_dir}/model_training_pl.py" '
        for u in params.keys():
            v = params[u]
            command_str += f"--{u} {v} " if not isinstance(v, bool) else f"--{u} "

        print(command_str)
        subprocess.run(command_str, shell=True)

        if n is not None and n > 1:
            self.perform_grid_search(search_space, n - 1)
        else:
            self.save_grid_search_results()

    def save_grid_search_results(self):
        df = self.find_best_models()
        best_model = df[df[f"val_{self.metric}"] == min(df[f"val_{self.metric}"])].iloc[0]
        best_model.to_json(f"{self.grid_search_dir}/best_parameters.json")

        # noinspection PyTypeChecker
        df.to_csv(f"{self.grid_search_dir}/grid_search({self.timestamp}).csv", index=False)


def aggregate_fold_histories(dir_: str):
    """Get history for each epoch and aggregate them (mean across epochs)."""
    histories = glob.glob(f"{dir_}/*/*/history.csv")

    df = pd.DataFrame()
    for history in histories:
        fold = history.split(os.sep)[-3][-1]
        df_fold = pd.read_csv(history)
        df_fold['fold'] = fold
        df = pd.concat([df, df_fold], ignore_index=True)

    df = df.groupby(by=["epoch"]).mean().reset_index()

    return df


def get_hyperparams(dir_: str):
    """Get hyperparameters."""
    files = list(Path(dir_).rglob("hparams.yaml"))

    with open(files[0], "r") as stream:
        hparams = yaml.load(stream, yaml.Loader)

    if "loss_weights" in hparams and isinstance(hparams["loss_weights"], tuple):
        weights = hparams.pop("loss_weights")
        hparams["weighted_loss"] = weights[-1]

    return hparams


if __name__ == "__main__":
    # Global variables
    project_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/"
    model_type = "Siamese_Baseline"
    keep_best_weights = False

    timestamp = datetime.now().strftime("%Y-%m-%d")
    # timestamp = '2022-01-03'
    grid_search_dir = f"{project_dir}/results/{model_type}{'_' if (len(model_type) > 0) else ''}grid_search({timestamp})/"

    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)

    hyperparams = {'lr': [1e-3, 1e-2, 1e-4],
                   "batch_size": [1, 16, 64, 128],
                   'adam': [True, False],
                   'momentum': [0.8, 0.9],
                   'weight_decay': [5e-4, 5e-3],
                   'weighted_loss': [0.5, 0.87],
                   'output_dim': [128, 256, 512],
                   'dropout_rate': [0, 0.25, 0.5],
                   }

    gridSearch = GridSearch(model_type, timestamp, grid_search_dir)
    gridSearch.perform_grid_search(hyperparams, n=8)
    gridSearch.save_grid_search_results()
