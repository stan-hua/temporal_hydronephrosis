import sys
import json
from collections import defaultdict
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from utilities.dataset_prep import load_dataset, get_data_dicts
from model_training_pl import parseArgs, modifyArgs

matplotlib.use("Qt5Agg")


def plot_loss(results_dir):
    """Visualize loss curves. Create subplots if there are multiple folds.

    @param results_dir: path to results directory
    """
    df = pd.read_csv(results_dir + "history.csv")

    # Create plot
    fig = plt.figure()
    if df.fold.nunique() == 5:
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(236)
        axs = [ax1, ax2, ax3, ax4, ax5]
    else:
        axs = [plt.subplot(111)]
    for i in df.fold.unique():
        sns.lineplot(data=df[df.fold == i], x="epoch", y="loss", hue="dset", ax=axs[i - 1])
        axs[i - 1].set_title(f"Fold {i}")

        if i == 4:
            axs[i - 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            axs[i - 1].get_legend().remove()
    plt.tight_layout()
    fig.suptitle("Train-Val Loss Curve")
    # Save figure
    plt.savefig(results_dir + "/loss.png", dpi=300)


class DataViewer:
    """DataViewer class. Used to describe sequences of images, and their class/covariate distribution.
    """

    def __init__(self, X, y, cov=None, X_test=None, y_test=None, cov_test=None):
        """
        Initialize DataViewer class.

        @param X: List containing sequence of images per patient (as numpy arrays)
        @param y: Class associated with each item in X
        @param cov: Covariate associated with each item in X
        """
        self.X = X
        self.X_test = X_test
        self.df_train = pd.DataFrame({"y": y})
        self.df_train["Num_visits"] = [len(x) for x in X] if (len(X.shape) != 4) else 1
        df_cov = pd.DataFrame(cov if not isinstance(cov, np.ndarray) else cov.tolist())
        self.df_train = pd.concat([self.df_train, df_cov], axis=1)

        if X_test is not None and y_test is not None:
            self.df_test = pd.DataFrame({"y": y_test})
            self.df_test["Num_visits"] = [len(x) for x in X_test] if (len(X_test.shape) != 4) else 1
            df_cov = pd.DataFrame(cov_test if not isinstance(cov_test, np.ndarray) else cov_test.tolist())
            self.df_test = pd.concat([self.df_test, df_cov], axis=1)

    def get_proportion_positive(self):
        """Returns proportion of samples that are positive."""
        return self.df["y"].mean()

    def show_num_visits(self, positive_only=False):
        """Show patient counts with n visits."""
        if positive_only:
            df_train = self.df_train[self.df_train['y'] == 1]
            df_test = self.df_test[self.df_test['y'] == 1] if self.X_test is not None else None
        else:
            df_train = self.df_train
            df_test = self.df_test

        counts_train = df_train.Num_visits.value_counts().reset_index()
        counts_train = counts_train.rename(columns={"index": "Number of Visits", "Num_visits": "Count"}).T

        print("==Training==:")
        print(counts_train)
        print("Percent Positives: ", np.mean(y_train_ == 1))
        df = counts_train.T
        print("# of Positive Examples with 2 or more visits: ", df[df["Number of Visits"] > 1]["Count"].sum(), "/",
              df["Count"].sum())

        counts_test = None
        if self.X_test is not None:
            counts_test = df_test.Num_visits.value_counts().reset_index()
            counts_test = counts_test.rename(columns={"index": "Number of Visits", "Num_visits": "Count"}).T

            print("==Validation/Test==:")
            print(counts_test)
            print("Percent Positives: ", np.mean(y_test_ == 1))
            df = counts_test.T
            print("# of Positive Examples with 2 or more visits: ", df[df["Number of Visits"] > 1]["Count"].sum(), "/",
                  df["Count"].sum())

        return counts_train, counts_test

    def show_num_patients(self):
        print("==Training Set==:")
        print("Total Num. of Patients: ", self.df_train.ID.nunique())
        print("Num. of Positive Patients: ", self.df_train[self.df_train.y == 1].ID.nunique())

        if self.X_test is not None:
            print("==Test Set==:")
            print("Total Num. of Patients: ", self.df_test.ID.nunique())
            print("Num. of Positive Patients: ", self.df_test[self.df_test.y == 1].ID.nunique())
            print("")

    def plot_cov_distribution(self):
        """Plot distribution of covariates in training and testing set."""

        def _plot_cov(df):
            return sns.pairplot(df[["Num_visits", "y", "Age_wks"]])

        if self.X_test is not None:
            fig = _plot_cov(self.df_train)
            # fig.set(title='Training Set')
            plt.title('Training Set')

            fig = _plot_cov(self.df_test)
            # fig.set(title='Test Set')
            plt.title('Test Set')
        else:
            fig = _plot_cov(self.df_train)
            # fig.set(title='Training Set')
            plt.title('Training Set')

    def plot_imaging_date_frequencies(self):
        """Plots histogram of imaging dates for positive & negative patients."""

        def _plot_frequencies(df, ax=None):
            ax1 = sns.histplot(data=df[df.y == 0], x="Imaging_Date", color="#55ACF8", bins=20, ax=ax, alpha=1)
            sns.histplot(data=df[df.y == 1], x="Imaging_Date", color="#F8A155", bins=20, ax=ax1, alpha=1)
            ax1.legend(["Negative", "Positive"])
            return ax1

        if self.X_test is not None:
            fig, axs = plt.subplots(nrows=1, ncols=2)

            _plot_frequencies(self.df_train, ax=axs[0])
            axs[0].set_title("Training Set")

            _plot_frequencies(self.df_test, ax=axs[1])
            axs[1].set_title("Testing Set")
        else:
            _plot_frequencies(self.df_train)
            plt.title("Training Set")


def load_data():
    args = parseArgs()
    modifyArgs(args)
    args.include_validation = False

    train_dict, test_dict = load_dataset(args.json_infile, test_prop=0.2, ordered_split=args.ordered_split,
                                         train_only=args.train_only,
                                         data_dir=args.data_dir)

    train_img_dict, train_label_dict, train_cov_dict, train_study_ids = get_data_dicts(train_dict,
                                                                                       data_dir=args.data_dir,
                                                                                       seq=not args.single_visit)
    test_img_dict, test_label_dict, test_cov_dict, test_study_ids = get_data_dicts(test_dict,
                                                                                   data_dir=args.data_dir,
                                                                                   seq=not args.single_visit,
                                                                                   last_visit_only=args.single_visit)

    train_set = train_img_dict, train_label_dict, train_cov_dict, train_study_ids
    test_set = test_img_dict, test_label_dict, test_cov_dict, test_study_ids
    return train_set, test_set


def describe_data(train_set, val_set):
    train_img_dict, train_label_dict, train_cov_dict, train_study_ids = train_set
    val_img_dict, val_label_dict, val_cov_dict, val_study_ids = val_set

    data_viewer = DataViewer(X_train_, y_train_, cov_train_,
                             X_val_, y_val_, cov_val_)
    # data_viewer.show_num_visits(positive_only=True)
    # data_viewer.show_num_patients()
    data_viewer.plot_cov_distribution()
    # data_viewer.plot_imaging_date_frequencies()


def process_cov_dict(cov_dict):
    df = pd.DataFrame(cov_dict).T.reset_index()
    df['ID'] = df['index'].map(lambda x: x.split('_')[0])
    df['Side_L'] = df['index'].map(lambda x: x.split('_')[1])
    df['ID_Side'] = df['index'].map(lambda x: '_'.join(x.split('_')[:2]))

    return df


def get_examples_info(df_cov, label_dict):
    def _extract_info_per_patient(df_):
        """Get the age range (max - min) of patient across all their visits, and get their number of visits"""
        age_range = max(df_["Age_wks"]) - min(df_['Age_wks'])
        num_visits = len(df_)
        id_ = df_['ID'].iloc[0]
        side = df_['Side_L'].iloc[0]

        row = pd.DataFrame({'age_range': age_range,
                            'num_visits': num_visits,
                            'ID': id_,
                            'side': side}, index=[0])
        return row

    df_examples = df_cov.groupby(by=['ID_Side']).apply(lambda df_: _extract_info_per_patient(df_))
    df_examples = df_examples.reset_index()
    df_examples.drop(columns=['level_1'], inplace=True)

    id_side_to_target = {"_".join(u.split("_")[:-1]): v for u, v in label_dict.items()}

    df_examples['surgery'] = df_examples['ID_Side'].map(lambda x: id_side_to_target[x])


if __name__ == "__main__":
    train_set, test_set = load_data()

    # describe_data((X_train_, y_train_, cov_train_), (X_test_, y_test_, cov_test_))

    # train_val_generator = make_validation_set(X_train_, y_train_, cov_train_, cv=True, num_folds=5)
    # i = 1
    # for train_val_fold in train_val_generator:  # only iterates once if not cross-fold validation
    #     print(f"Fold {i}/{5}")
    #     X_train, y_train, cov_train, X_val, y_val, cov_val = train_val_fold
    #     describe_data((X_train, y_train, cov_train), (X_val, y_val, cov_val))
    #     i += 1
