"""
Use this module to prepare data for model training. Can be used to prepare sequence of images.
"""
from collections import defaultdict
from functools import partial
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler


# ==DATA LOADING==:
class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X: list, y: list, cov: list):
        """If <include_cov>, each item becomes a tuple of (dictionary of img X and covariates, target label). Otherwise,
        each item is a tuple of (image X, target label)."""
        self.X = [torch.from_numpy(e).float() for e in X]
        self.y = y
        self.cov = cov

    def __getitem__(self, index):
        imgs, target, cov = self.X[index], self.y[index], self.cov[index]

        return imgs, target, cov

    def __len__(self):
        return len(self.X)

    def get_class_proportions(self):
        """Returns an array of values inversely proportional to the number of items in the class. Classes with a greater
        number of examples compared to other classes receives lesser weight.
        """
        counts = np.bincount(self.y)
        labels_weights = 1. / counts
        weights = labels_weights[self.y]
        return weights


def recreate_train_test_split(X_train, y_train, cov_train,
                              X_test, y_test, cov_test):
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    cov = np.concatenate([cov_train, cov_test])

    data_by_patients = group_data_by_ID(X, y, cov)
    data_by_patients = shuffle(data_by_patients, random_state=1)
    train_data_, test_data_ = split(data_by_patients, 0.3)


def make_validation_set(X_train, y_train, cov_train,
                        cv=False, num_folds=5,
                        train_val_split=0.2):
    """HELPER FUNCTION. Split training data into training and validation splits. If cross-fold validation specified,
    return generator of <num_folds> train-val splits.

    If include_validation is false, then return the same input.
    """

    def split(data, split):
        return data[:-int(len(data) * split)], data[-int(len(data) * split):]

    if not cv:
        X_train, X_val = split(X_train, train_val_split)
        y_train, y_val = split(y_train, train_val_split)
        cov_train, cov_val = split(cov_train, train_val_split)
        return [(X_train, y_train, cov_train, X_val, y_val, cov_val)]
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        return ((X_train[train_index], y_train[train_index], cov_train[train_index],
                 X_train[test_index], y_train[test_index], cov_train[test_index]) for train_index, test_index in
                skf.split(X_train, y_train))


def create_data_loaders(X_train, y_train, cov_train, X_val, y_val, cov_val, X_test, y_test, cov_test,
                        args, params, val_test_params):
    # Datasets
    training_set = KidneyDataset(X_train, y_train, cov_train)
    val_set = KidneyDataset(X_val, y_val, cov_val) if (X_val is not None) else None
    test_set = KidneyDataset(X_test, y_test, cov_test)

    # Weighted sampling
    if args.balance_classes:
        samples_weight = torch.from_numpy(training_set.get_class_proportions())
        print(samples_weight)
        sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), len(training_set))
        params["shuffle"] = False
        params["sampler"] = sampler

    # Data Loaders
    training_generator = DataLoader(training_set,
                                    collate_fn=pad_collate if args.standardize_seq_length else None,
                                    **params)
    val_generator = DataLoader(val_set,
                               collate_fn=pad_collate if args.standardize_seq_length else None,
                               **val_test_params) if (X_val is not None) else None

    test_generator = DataLoader(test_set,
                                collate_fn=pad_collate if args.standardize_seq_length else None,
                                **val_test_params)

    return training_generator, val_generator, test_generator


# ==HELPER FUNCTIONS==:
def pad_collate(batch):
    """Pad batch to be of the longest sequence length.

    @returns tuple containing (padded X, y, cov and length of each sequence in batch)
    """
    (x_t, y_t, cov_t) = zip(*batch)

    x_lens = [len(x) for x in x_t]
    x_pad = pad_sequence(x_t, batch_first=True, padding_value=0)

    # Perform padding in-place (for target and covariates)
    for y in y_t:
        if isinstance(y, list) and (len(y) != max(x_lens)):     # zero-padding
            y.extend([0] * abs(len(y) - max(x_lens)))
            assert len(y) == max(x_lens)

    for cov in cov_t:
        if len(cov) != max(x_lens):     # pad with empty dictionary
            cov.extend([{}] * abs(len(cov) - max(x_lens)))
            assert len(cov) == max(x_lens)

    return (x_pad, np.array(x_lens)), y_t, cov_t


def prepare_data_into_sequences(X_train, y_train, cov_train,
                                X_test, y_test, cov_test,
                                single_visit, single_target):
    """Prepare data into sequences of (pairs of images)."""

    def sort_data_by_ID(t_x, t_y, t_cov):
        """Sort by patient IDs."""
        # cov_train_, X_train_, y_train_ = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        cov_train_, X_train_, y_train_ = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: x[0]["ID"]))
        return X_train_, y_train_, cov_train_

    def group(t_x, t_y, t_cov, include_side=True):
        """Group images according to patient ID. If <include_side>, groups by patient ID and by kidney side.
        And sort by date.
        """
        x, y, cov = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(t_cov)):
            if include_side:  # split data per kidney e.g 5.0Left, 5.0Right, 6.0Left, ...
                # id_ = t_cov[i].split("_")[0] + t_cov[i].split("_")[4]
                id_ = f"{t_cov[i]['ID']}_{t_cov[i]['Side_L']}"
            else:  # split only on id e.g. 5.0, 6.0, 7.0, ...
                # id_ = t_cov[i].split("_")[0]
                id_ = str(t_cov[i]['ID'])
            x[id_].append(t_x[i])
            y[id_].append(t_y[i])
            cov[id_].append(t_cov[i])

        # Sort by date
        for id_ in x.keys():
            x[id_], y[id_], cov[id_] = zip(*sorted(zip(x[id_], y[id_], cov[id_]),
                                                   key=lambda p: p[2]["Imaging_Date"]))
                                                   # key=lambda p: datetime.strptime(p[2].split("_")[5], "%Y-%m-%d")))

        # Convert to numpy array
        X_grouped = np.asarray([np.asarray(e) for e in list(x.values())])
        return X_grouped, np.asarray(list(y.values())), np.asarray(list(cov.values()))

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

    X_train, y_train, cov_train = sort_data_by_ID(X_train, y_train, cov_train)
    X_test, y_test, cov_test = sort_data_by_ID(X_test, y_test, cov_test)

    if not single_visit:  # group images by patient ID and sort by date
        X_train, y_train, cov_train = group(X_train, y_train, cov_train)
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)

        if single_target:
            y_train = np.array([seq[-1] for seq in y_train])
            y_test = np.array([seq[-1] for seq in y_test])

    if single_visit:  # only test on last visit
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)
        X_test, y_test, cov_test = get_only_last_visits(X_test, y_test, cov_test)

        # if single_target:   # notice that test already has only 1 y-value
        #     y_train = np.array([seq[-1] for seq in y_train])
        print(len(X_train), len(y_train))

    return np.array(X_train), np.array(y_train), np.array(cov_train), np.array(X_test), np.array(y_test), np.array(
        cov_test)


def parse_cov(cov: str, side=True, age=True, sex=False) -> dict:
    """HELPER FUNCTION. Given a string, extract specified covariates. By default, extracts kidney side and
    age (in weeks)."""
    # If already a dictionary, simply return the same dictionary
    if isinstance(cov, dict):
        return cov

    cov_dict = {}
    cov_split = cov.split("_")

    try:
        cov_dict["Imaging_Date"] = datetime.strptime(cov_split[6], "%Y-%m-%d")
    except ValueError:
        # TODO: Remove?
        cov_dict["Imaging_Date"] = datetime.now()

    cov_dict["ID"] = float(cov_split[0])

    if age:
        cov_dict["Age_wks"] = float(cov_split[1])
    if side:
        cov_dict["Side_L"] = 1 if cov_split[4] == 'Left' else 0
    if sex:
        cov_dict["Sex"] = 1 if cov_split[2] == "M" else "F"

    return cov_dict


def remove_unnecessary_cov(cov):
    if isinstance(cov, dict):
        cov.pop("ID")
        cov.pop("Imaging_Date")
    else:
        try:
            for c in cov:
                remove_unnecessary_cov(c)
        except TypeError:
            pass
