"""
Use this module to prepare data for model training. Can be used to prepare sequence of images.
"""
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler


class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, cov):
        # self.X = [torch.tensor(e, requires_grad=True).float() for e in X]
        self.X = [torch.from_numpy(e).float() for e in X]
        # self.X = torch.from_numpy(X).float()
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


def pad_collate(batch):
    """Pad batch to be of the longest sequence length.

    @returns tuple containing (padded X, y, cov and length of each sequence in batch)
    """
    (x_t, y_t, cov_t) = zip(*batch)
    x_lens = [len(x) for x in x_t]

    x_pad = pad_sequence(x_t, batch_first=True, padding_value=0)

    y_pad = []
    for y in y_t:
        if isinstance(y, list) or isinstance(y, tuple):
            y_new = y.copy()
            if len(y) != max(x_lens):
                # TODO: What to pad with?
                y_new.extend([""] * abs(len(y) - max(x_lens)))
            y_pad.append(y_new)
        else:  # single target
            y_pad.append(y)

    cov_pad = []
    for cov in cov_t:
        cov_new = cov.copy()
        if len(cov) != max(x_lens):
            # TODO: What to pad with?
            cov_new.extend([""] * abs(len(cov) - max(x_lens)))
        cov_pad.append(cov_new)

    return (x_pad, np.array(x_lens)), y_pad, cov_pad


def prepare_data_into_sequences(X_train, y_train, cov_train,
                                X_test, y_test, cov_test,
                                single_visit, single_target, fix_seq_length):
    """Prepare data into sequences of (pairs of images)"""

    def sort_data(t_x, t_y, t_cov):
        cov_train_, X_train_, y_train_ = zip(*sorted(zip(t_cov, t_x, t_y), key=lambda x: float(x[0].split("_")[0])))
        return X_train_, y_train_, cov_train_

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
            pass
            # X_train, y_train, cov_train = standardize_seq_length(X_train, y_train, cov_train)
            # X_test, y_test, cov_test = standardize_seq_length(X_test, y_test, cov_test)

    if single_visit:  # only test on last visit
        X_test, y_test, cov_test = group(X_test, y_test, cov_test)
        X_test, y_test, cov_test = get_only_last_visits(X_test, y_test, cov_test)

        # if single_target:   # notice that test already has only 1 y-value
        #     y_train = np.array([seq[-1] for seq in y_train])
        print(len(X_train), len(y_train))

    return np.array(X_train), np.array(y_train), np.array(cov_train), np.array(X_test), np.array(y_test), np.array(
        cov_test)


def make_validation_set(X_train, y_train, cov_train, cv=False, num_folds=5):
    """HELPER FUNCTION. Split training data into training and validation splits. If cross-fold validation specified,
    return generator of <num_folds> train-val splits.

    If include_validation is false, then return the same input.
    """

    def split(train_, split_=0.2):
        """Save last person as a validation test?"""
        return train_[:-int(len(train_) * split_)], train_[-int(len(train_) * split_):]

    if not cv:
        X_train, X_val = split(X_train)
        y_train, y_val = split(y_train)
        cov_train, cov_val = split(cov_train)
        return [(X_train, y_train, cov_train, X_val, y_val, cov_val)]
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        return ((X_train[train_index], y_train[train_index], cov_train[train_index],
                 X_train[test_index], y_train[test_index], cov_train[test_index]) for train_index, test_index in
                skf.split(X_train, y_train))


def create_data_loaders(X_train, y_train, cov_train, X_val, y_val, cov_val, X_test, y_test, cov_test,
                        args,
                        params, val_test_params):
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
