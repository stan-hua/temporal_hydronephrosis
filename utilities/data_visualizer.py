import sys
import json
from collections import defaultdict
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from utilities.dataset_prep import prepare_data_into_sequences, parse_cov

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
        self.df_train = pd.DataFrame({"length": [len(x) for x in X], "y": y})
        self.df_train["cov"] = cov

        if X_test is not None and y_test is not None:
            self.df_test = pd.DataFrame({"length": [len(x) for x in X_test], "y": y_test})
            self.df_test["cov"] = cov_test

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

        counts_train = df_train.length.value_counts().reset_index()
        counts_train = counts_train.rename(columns={"index": "Number of Visits", "length": "Count"}).T

        counts_test = None

        if self.X_test is not None:
            counts_test = df_test.length.value_counts().reset_index()
            counts_test = counts_test.rename(columns={"index": "Number of Visits", "length": "Count"}).T

        return counts_train, counts_test


def load_data():
    git_dir = "C:/Users/Stanley Hua/projects/"
    sys.path.insert(0, git_dir + '/nephronetwork/0.Preprocess/')
    sys.path.insert(0, git_dir + '/nephronetwork/1.Models/siamese_network/')
    from load_dataset_LE import load_dataset

    # Load data
    X_train_, y_train_, cov_train_, X_test_, y_test_, cov_test_ = load_dataset(
        views_to_get="siamese",
        sort_by_date=True,
        pickle_file=git_dir + "nephronetwork/0.Preprocess/preprocessed_images_20190617.pickle",
        contrast=1,
        split=0.7,
        get_cov=True,
        bottom_cut=0,
        etiology="B",
        crop=0,
        git_dir=git_dir
    )

    if isinstance(cov_train_, list) and isinstance(cov_test_, list):
        func_parse_cov = partial(parse_cov, age=True, side=True, sex=False)
        cov_train_ = list(map(func_parse_cov, cov_train_))
        cov_test_ = lists(map(func_parse_cov, cov_test_))

    return X_train_, y_train_, cov_train_, X_test_, y_test_, cov_test_


if __name__ == "__main__":
    X_train_, y_train_, cov_train_, X_test_, y_test_, cov_test_ = load_data()

    X_train_, y_train_, cov_train_, X_test_, y_test_, cov_test_ = prepare_data_into_sequences(X_train_, y_train_,
                                                                                              cov_train_,
                                                                                              X_test_, y_test_,
                                                                                              cov_test_,
                                                                                              single_visit=False,
                                                                                              single_target=True)

    X = np.concatenate([X_train_, X_test_])
    y = np.concatenate([y_train_, y_test_])
    cov = np.concatenate([cov_train_, cov_test_])

    def group_data_by_ID(t_x, t_y, t_cov):
        """Group by patient IDs."""
        ids = [cov[0]["ID"] for cov in t_cov]
        data_by_id = {id_: [] for id_ in ids}

        for i in range(len(t_x)):
            id_ = t_cov[i][0]["ID"]
            data_by_id[id_].append((t_x[i], t_y[i], t_cov[i]))

        return list(data_by_id.values())

    data_by_patients = group_data_by_ID(X, y, cov)
    data_by_patients = shuffle(data_by_patients, random_state=1)
    train_data_, test_data_ = split(data_by_patients, 0.3)

    train_data = []
    for d in train_data_:
        train_data.extend(d)
    test_data = []
    for d in test_data_:
        test_data.extend(d)

    X_train_, y_train_, cov_train_ = zip(*train_data)
    X_test_, y_test_, cov_test_ = zip(*test_data)

    X_train_, y_train_, cov_train_ = np.array(X_train_, dtype=object), np.array(y_train_, dtype=object), np.array(cov_train_, dtype=object)
    X_test_, y_test_, cov_test_ = np.array(X_test_, dtype=object), np.array(y_test_, dtype=object), np.array(cov_test_, dtype=object)

    data_viewer = DataViewer(X_train_, y_train_, cov_train_,
                             X_test_, y_test_, cov_test_)
    counts_train, counts_test = data_viewer.show_num_visits(positive_only=True)

    print(counts_train)
    print("Percent Positives: ", np.mean(y_train_ == 1))
    df = counts_train.T
    print("# of Positive Patients with 2 or more visits: ", df[df["Number of Visits"] > 1]["Count"].sum(), "/", df["Count"].sum())

    print(counts_test)
    print("Percent Positives: ", np.mean(y_test_ == 1))
    df = counts_test.T
    print("# of Positive Patients with 2 or more visits: ", df[df["Number of Visits"] > 1]["Count"].sum(), "/", df["Count"].sum())
