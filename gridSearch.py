"""Perform Randomized Grid Search for hyperparameters.
"""
import os
import subprocess
import shutil
from datetime import datetime
from glob import glob

import pandas as pd

project_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/"
model_type = "ConvPooling"
keep_best_weights = True


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
        except:
            pass
        return [file for file in glob(f"{self.grid_search_dir}/*") if "csv" not in file]

    @staticmethod
    def findBestEpoch(df):
        """Returns filtered dataframe for epochs with best validation set AUPRC."""
        if df.fold.nunique() == 1:
            df_val = df[df.dset == "val"]
            best_val_auprc = df_val["auprc"].max()
            best_val_epoch = df_val[df_val.auprc == best_val_auprc]

            return best_val_epoch
        else:
            folds_best_val_epoch = []
            for fold in df.fold.unique().tolist():
                df_ = df[df.fold == fold]
                df_val = df_[df_.dset == "val"]
                best_val_auprc = df_val["auprc"].max()
                folds_best_val_epoch.append(df_val[df_val.auprc == best_val_auprc])
            kfold_best_val_results = pd.concat(folds_best_val_epoch)
            return kfold_best_val_results

    def findBestPerformingModel(self):
        """.
        :param timestamp: timestamp when models were run
        :return
        """
        result_directories = self.getAllTrainingDirectories()

        df_accum = pd.DataFrame()
        for dir_ in result_directories:
            # Get epoch for best results and model parameters
            df_results = pd.read_csv(f"{dir_}/history.csv")
            df_best_epoch = findBestEpoch(df_results)
            params = pd.read_csv(f"{dir_}/info.csv").iloc[0].to_dict()

            # If KFold, average results over all folds. Also, remove all weights or keep best weights if specified (only
            # for train-val split).
            if len(df_best_epoch) > 1:
                num_fold = len(df_best_epoch)
                df_best_epoch = df_best_epoch.mean().to_frame().T
                df_best_epoch["fold"] = num_fold
                df_best_epoch["dset"] = "val"
                removeUnnecessaryWeights(dir_, -10)
            else:
                removeUnnecessaryWeights(dir_, df_best_epoch.epoch.iloc[0] if keep_best_weights else -10)

            for key in params:
                df_best_epoch[key] = params[key]
            df_best_epoch["dir"] = dir_

            df_accum = pd.concat([df_accum, df_best_epoch])

        print(df_accum)

        return df_accum

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

    def perform_grid_search(self,
                            lrs: list,
                            batch_sizes: list,
                            momentums: list,
                            adams: list,
                            num_epochs: int):
        """
        Performs grid search on given specified parameter lists.
        :param lrs: learning rates to test
        :param batch_sizes: batch sizes to test
        :param momentums: SGD momentums to test (if adam is True)
        :param adams: If contains True, will run adam. If contains False, will run SGD.
        :param num_epochs: number of epochs
        """
        for batch_size in batch_sizes:
            for lr in lrs:
                for adam in adams:
                    for momentum in momentums:
                        if adam and len(momentums) > 1 and momentum in momentums[1:]:
                            continue

                        subprocess.run(f'python "{project_dir}/modelTraining.py" '
                                       f"--batch_size {batch_size} "
                                       f"--lr {lr} "
                                       f"--stop_epoch {num_epochs} "
                                       f"--momentum {momentum} "
                                       + (f"--adam" if adam else ""),
                                       shell=True)

        df = findBestPerformingModel(timestamp)
        # noinspection PyTypeChecker
        df.to_csv(f"{self.grid_search_dir}/grid_search({self.timestamp}).csv", index=False)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp = "2021-11-26"
    grid_search_dir = f"{project_dir}/results/{model_type}{'_' if (len(model_type) > 0) else ''}grid_search({timestamp})/"

    if not os.path.exists(grid_search_dir):
        os.mkdir(grid_search_dir)

    lrs = [0.0001, 0.001]  # 0.00001 0.0001, 0.001
    batch_sizes = [5]  # 15
    momentums = [0.85]  # 0.9, 0.95
    adams = [False]
    num_epochs = 100

    gridSearch = GridSearch(model_type, timestamp, grid_search_dir)
