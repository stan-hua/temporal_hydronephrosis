"""Perform Randomized Grid Search for hyperparameters.
"""
import os
import subprocess
import shutil
from datetime import datetime
from glob import glob

import pandas as pd


class GridSearch:
    timestamp_: str
    grid_search_dir_: str
    model_type_: str

    def __init__(self, model_type_, timestamp_, grid_search_dir_):
        self.model_type = model_type_
        self.timestamp = timestamp_
        self.grid_search_dir = grid_search_dir_

    def getAllTrainingDirectories(self):
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

    @staticmethod
    def findBestEpoch(df, metric="loss"):
        """Returns epoch with validation with minimum <metric> value. If cross-fold validation, first average over
        folds
        """
        df_val = df[df.dset == "val"]
        df_mean_val = df_val.groupby(by=["epoch"]).mean()
        df_best_epoch = df_mean_val[df_mean_val[metric] == min(df_mean_val[metric])]
        df_best_epoch["dset"] = "val"

        return df_best_epoch

    def findBestPerformingModels(self):
        """If cross-fold validation done, average over epochs. Get row with the lowest validation loss.

        @return dataframe containing the best validation set results, where each row corresponds to a tested model 
                hyperparameters
        """
        result_directories = self.getAllTrainingDirectories()

        df_accum = pd.DataFrame()
        for dir_ in result_directories:
            # Get epoch for best results and model parameters
            df_results = pd.read_csv(f"{dir_}/history.csv")
            df_best_epoch = self.findBestEpoch(df_results, "loss")
            self.removeUnnecessaryWeights(dir_, df_best_epoch.epoch.iloc[0] if keep_best_weights else -10)

            # Add in parameters
            params = pd.read_csv(f"{dir_}/info.csv").iloc[0].to_dict()
            for key in params:
                df_best_epoch[key] = params[key]
            df_best_epoch["dir"] = dir_
            df_accum = pd.concat([df_accum, df_best_epoch])

        print(df_accum)

        return df_accum

    def saveBestParameters(self, df):
        """Given dataframe where each row is a model run, extract the best performing model based on the validation set
        and save hyperparameters."""
        df[df.sum_auc == df.sum_auc.max()].iloc[0].to_json(f"{self.grid_search_dir}/best_parameters.json")

    @staticmethod
    def removeUnnecessaryWeights(directory, best_epoch=-10):
        """Remove all weights in directory for model epochs 2 above and/or below best."""
        if not os.path.exists(directory):
            return

        all_weights_paths = glob(f"{directory}/model-epoch_*.pth")
        if len(all_weights_paths) > 0:
            epoch_numbers = [int(path.split("_")[-1].split(".")[0][0]) for path in all_weights_paths]
            distance = [abs(epoch - best_epoch) for epoch in epoch_numbers]
            closest_epoch_index = [i for i in range(len(distance)) if distance[i] == min(distance)][0]

            for i in range(len(epoch_numbers)):
                if i not in [closest_epoch_index - j for j in [-1, 0, 1]]:
                    os.remove(all_weights_paths[i])

    @staticmethod
    def perform_grid_search(lrs_: list,
                            batch_sizes_: list,
                            momentums_: list,
                            adams_: list,
                            num_epochs_: int):
        """
        Performs grid search on given specified parameter lists.
        @param lrs_: learning rates to test
        @param batch_sizes_: batch sizes to test
        @param momentums_: SGD momentums to test (if adam is True)
        @param adams_: If contains True, will run adam. If contains False, will run SGD.
        @param num_epochs_: number of epochs
        """
        for batch_size in batch_sizes_:
            for lr in lrs_:
                for adam in adams_:
                    for momentum in momentums_:
                        subprocess.run(f'python "{project_dir}/model_training.py" '
                                       f"--batch_size {batch_size} "
                                       f"--lr {lr} "
                                       f"--stop_epoch {num_epochs_} "
                                       f"--momentum {momentum} "
                                       + (f"--adam" if adam else ""),
                                       shell=True)

    def save_grid_search_results(self):
        df = self.findBestPerformingModels()
        self.saveBestParameters(df)
        # noinspection PyTypeChecker
        df.to_csv(f"{self.grid_search_dir}/grid_search({self.timestamp}).csv", index=False)


def delete_all_weights():
    for file_name in glob.glob(f"{project_dir}results/*/*/*.pth"):
        print(file_name)
        os.remove(file_name)


if __name__ == "__main__":
    # Global variables
    project_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/"
    model_type = "LSTM"
    keep_best_weights = False

    timestamp = datetime.now().strftime("%Y-%m-%d")
    grid_search_dir = f"{project_dir}/results/{model_type}{'_' if (len(model_type) > 0) else ''}grid_search({timestamp})/"

    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)

    lrs = []  # 0.00001, 0.0001, 0.001
    batch_sizes = [1]  # 6, 12
    momentums = [0.85, 0.95]  # 0.9, 0.95
    adams = [True]
    num_epochs = 100

    gridSearch = GridSearch(model_type, timestamp, grid_search_dir)
    gridSearch.perform_grid_search(lrs, batch_sizes, momentums, adams, num_epochs)
    gridSearch.save_grid_search_results()
