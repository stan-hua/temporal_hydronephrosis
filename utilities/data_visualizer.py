import sys
import json
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

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

    def preprocess_covariates(self):
        """Preprocess covariates and update dataframe."""
        gender = self.df_train['cov'].map(lambda lst: [i.split("_")[2] for i in lst][0])
        machine = self.df_train['cov'].map(lambda lst: [i.split("_")[-1] for i in lst])
        date = self.df_train['cov'].map(lambda lst: [i.split("_")[-2] for i in lst])

    def show_num_visits(self):
        """Show patient counts with n visits."""
        counts_train = self.df_train.length.value_counts().reset_index()
        counts_train = counts_train.rename(columns={"index": "Number of Visits", "length": "Count"}).T

        counts_test = None

        if self.X_test is not None:
            counts_test = self.df_test.length.value_counts().reset_index()
            counts_test = counts_test.rename(columns={"index": "Number of Visits", "length": "Count"}).T

        return counts_train, counts_test


def prepare_data_into_sequences(X_train, y_train, cov_train,
                                X_test, y_test, cov_test,
                                single_visit, single_target, fix_seq_length):
    """Prepare data into sequences of (pairs of images)"""

    def sort_data(t_x, t_y, t_cov):
        cov, X, y = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return X, y, cov

    def group(t_x, t_y, t_cov):
        """Group images according to patient ID"""
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            # split data per kidney e.g 5.0Left, 5.0Right, 6.0Left, ...
            id_ = t_cov[i].split("_")[0] + t_cov[i].split("_")[4]
            # id = t_cov[i].split("_")[0] # split only on id e.g. 5.0, 6.0, 7.0, ...
            x[id_].append(t_x[i])
            y[id_].append(t_y[i])
            cov[id_].append(t_cov[i])
        # convert to np array
        organized_X_train = np.asarray([np.asarray(e) for e in list(x.values())])
        return organized_X_train, np.asarray(list(y.values())), np.asarray(list(cov.values()))

    def get_only_last_visits(t_x, t_y, t_cov):
        """Slice data to get only latest n visits."""
        x, y, cov = [], [], []
        for i, e in enumerate(t_x):
            curr_x = e[-1:]
            curr_x = curr_x.transpose((1, 0, 2, 3))
            curr_x = curr_x.squeeze()
            x.append(curr_x)
            y.append(t_y[i][-1])
            cov.append(t_cov[i][-1])
        return np.asarray(x, dtype=np.float64), y, cov

    def standardize_seq_length(X_t, y_t, cov_t):
        """Zero pad batch of varying sequence length to the max length.

        ==Precondition==:
            - Input is already grouped by patient ID.
        """
        longest_seq = max([len(x) for x in X_t])
        X_pad = []
        for x in X_t:
            x_new = np.zeros((longest_seq, 2, 256, 256))
            x_new[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
            X_pad.append(x_new)

        y_pad = []
        for y in y_t:
            if isinstance(y, list) or isinstance(y, tuple):
                y_new = y.copy()
                if len(y) != longest_seq:
                    y_new.extend([""] * abs(len(y) - longest_seq))
                y_pad.append(y_new)
            else:  # single target
                y_pad.append(y)

        cov_pad = []
        for cov in cov_t:
            cov_new = cov.copy()
            if len(cov) != longest_seq:
                cov_new.extend([""] * abs(len(cov) - longest_seq))
            cov_pad.append(cov_new)

        # x_t_pad = pad_sequence(x_t, batch_first=True, padding_value=0)
        # if len(y_t) > 1:
        #     y_t_pad = pad_sequence([torch.from_numpy(y) for y in y_t], batch_first=True, padding_value=0)
        # else:
        #     y_t_pad = y_t

        return X_pad, y_pad, cov_pad

    X_train, y_train, cov_train = sort_data(X_train, y_train, cov_train)
    X_test, y_test, cov_test = sort_data(X_test, y_test, cov_test)

    if not single_visit:  # group images by patient ID
        X_train, y_train, cov_train = group(X_train, y_train, cov_train)
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)

        if single_target:
            y_train = np.array([seq[-1] for seq in y_train])
            y_test = np.array([seq[-1] for seq in y_test])

        if fix_seq_length:
            X_train, y_train, cov_train = standardize_seq_length(X_train, y_train, cov_train)
            X_test, y_test, cov_test = standardize_seq_length(X_test, y_test, cov_test)

    if single_visit:  # only test on last visit
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)
        X_test, y_test, cov_test = get_only_last_visits(X_test, y_test, cov_test)

        # if single_target:   # notice that test already has only 1 y-value
        #     y_train = np.array([seq[-1] for seq in y_train])
        print(len(X_train), len(y_train))

    return np.array(X_train), np.array(y_train), np.array(cov_train), np.array(X_test), np.array(y_test), np.array(
        cov_test)


if __name__ == "__main__":
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

    X_train_, y_train_, cov_train_, X_test_, y_test_, cov_test_ = prepare_data_into_sequences(X_train_, y_train_, cov_train_,
                                                                                              X_test_, y_test_, cov_test_,
                                                                                              single_visit=False,
                                                                                              single_target=True,
                                                                                              fix_seq_length=False
                                                                                              )
