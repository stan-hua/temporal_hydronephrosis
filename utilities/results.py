import json
import os
import sys

import pandas as pd
import torch


class Results:
    """Stores results from model training and evaluation (on validation & testing set) per epoch.
    Also used to save model weights.
    """

    def __init__(self, y_train, y_val, y_test, args=None, fold=-1,
                 results_summary_path="", auc_path=""):
        self.results_summary_path = results_summary_path
        self.auc_path = auc_path

        self.epoch = -1
        self.fold = fold  # current cross-fold split
        self.dsets = []
        self.args = args

        self.curr_val_loss = 10000
        self.train_val_auprc = 0, 0

        self.all_targets_train, self.all_targets_val, self.all_targets_test = [], [], []

        self.accurate_pred_train, self.accurate_pred_val, self.accurate_pred_test = 0, 0, 0
        self.loss_accum_train, self.loss_accum_val, self.loss_accum_test = 0, 0, 0

        self.all_pred_prob_train, self.all_pred_prob_val, self.all_pred_prob_test = [], [], []
        self.all_pred_label_train, self.all_pred_label_val, self.all_pred_label_test = [], [], []

        self.all_patient_ID_train, self.all_patient_ID_val, self.all_patient_ID_test = [], [], []

        self.counter_train, self.counter_val, self.counter_test = 0, 0, 0
        try:
            self.totalTrainItems, self.totalValItems, self.totalTestItems = (sum(len(seq) for seq in dset) for dset in
                                                                             [y_train, y_val, y_test])
        except TypeError:
            self.totalTrainItems, self.totalValItems, self.totalTestItems = [len(y_train), len(y_val), len(y_test)]

        self.all_targets_train_tensor, self.all_targets_val_tensor, self.all_targets_test_tensor = None, None, None
        self.all_pred_prob_train_tensor, self.all_pred_prob_val_tensor, self.all_pred_prob_test_tensor = None, None, None
        self.all_pred_label_train_tensor, self.all_pred_label_val_tensor, self.all_pred_label_test_tensor = None, None, None

    def reset(self):
        """Reset stored variables for next epoch."""
        self.all_targets_train, self.all_targets_val, self.all_targets_test = [], [], []

        self.accurate_pred_train, self.accurate_pred_val, self.accurate_pred_test = 0, 0, 0
        self.loss_accum_train, self.loss_accum_val, self.loss_accum_test = 0, 0, 0

        self.all_pred_prob_train, self.all_pred_prob_val, self.all_pred_prob_test = [], [], []
        self.all_pred_label_train, self.all_pred_label_val, self.all_pred_label_test = [], [], []

        self.all_patient_ID_train, self.all_patient_ID_val, self.all_patient_ID_test = [], [], []

        self.counter_train, self.counter_val, self.counter_test = 0, 0, 0

        self.all_targets_train_tensor, self.all_targets_val_tensor, self.all_targets_test_tensor = None, None, None
        self.all_pred_prob_train_tensor, self.all_pred_prob_val_tensor, self.all_pred_prob_test_tensor = None, None, None
        self.all_pred_label_train_tensor, self.all_pred_label_val_tensor, self.all_pred_label_test_tensor = None, None, None

    def concat_results(self):
        """HELPER FUNCTION. Concatenates targets, predicted label and prediction probabilities."""
        if "train" in self.dsets:
            self.all_targets_train_tensor = torch.cat(self.all_targets_train)
            self.all_pred_prob_train_tensor = torch.cat(self.all_pred_prob_train)
            self.all_pred_label_train_tensor = torch.cat(self.all_pred_label_train)

        if "val" in self.dsets:
            self.all_targets_val_tensor = torch.cat(self.all_targets_val)
            self.all_pred_prob_val_tensor = torch.cat(self.all_pred_prob_val)
            self.all_pred_label_val_tensor = torch.cat(self.all_pred_label_val)

        if "test" in self.dsets:
            self.all_targets_test_tensor = torch.cat(self.all_targets_test)
            self.all_pred_prob_test_tensor = torch.cat(self.all_pred_prob_test)
            self.all_pred_label_test_tensor = torch.cat(self.all_pred_label_test)

    def verify_length(self, y_train, y_val, y_test):
        """Verifies that the lengths of sequences in training/val/test set match predictions and labels."""
        if "train" in self.dsets:
            if not self.args.single_target:
                assert len(self.all_pred_prob_train_tensor) == self.totalTrainItems
                assert len(self.all_pred_label_train_tensor) == self.totalTrainItems
                assert len(self.all_targets_train_tensor) == self.totalTrainItems

            if not self.args.single_visit:
                pass
                # assert len(self.all_patient_ID_train) == len(y_train)

        if "val" in self.dsets:
            if not self.args.single_target:
                assert len(self.all_pred_prob_val_tensor) == self.totalValItems
                assert len(self.all_pred_label_val_tensor) == self.totalValItems
                assert len(self.all_targets_val_tensor) == self.totalValItems

            if not self.args.single_visit:
                pass
                # assert len(self.all_patient_ID_val) == len(y_val)

        if "test" in self.dsets:
            if not self.args.single_target:
                assert len(self.all_pred_prob_test_tensor) == self.totalTestItems
                assert len(self.all_pred_label_test_tensor) == self.totalTestItems
                assert len(self.all_targets_test_tensor) == self.totalTestItems

            if not self.args.single_visit:
                pass
                # assert len(self.all_patient_ID_test) == len(y_test)

    def get_dset_results(self, dset):
        """HELPER FUNCTION. Return results for specified dset (train/val/test)."""
        if dset == "train":
            y_score = self.all_pred_prob_train_tensor.cpu().detach().numpy()
            y_true = self.all_targets_train_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_train_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_train) / self.counter_train
            loss = self.loss_accum_train / self.counter_train
        elif dset == "val":
            y_score = self.all_pred_prob_val_tensor.cpu().detach().numpy()
            y_true = self.all_targets_val_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_val_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_val) / self.counter_val
            loss = self.loss_accum_val / self.counter_val
        else:
            y_score = self.all_pred_prob_test_tensor.cpu().detach().numpy()
            y_true = self.all_targets_test_tensor.cpu().detach().numpy()
            y_pred = self.all_pred_label_test_tensor.cpu().detach().numpy()
            acc = int(self.accurate_pred_test) / self.counter_test
            loss = self.loss_accum_test / self.counter_test

        return y_score, y_true, y_pred, acc, loss

    def preprocess_results(self):
        """HELPER FUNCTION. Process results on train/val/test sets from model at current epoch and return as dataframe
        (containing accuracy, loss, etc.) and dictionary auc details.
        """
        sys.path.insert(0, self.args.git_dir + '/nephronetwork/2.Results/')
        from process_results import get_metrics

        df_results = pd.DataFrame()
        auc_dict = {}
        for dset in self.dsets:
            y_score, y_true, y_pred, acc, loss = self.get_dset_results(dset)
            results = get_metrics(y_score=y_score, y_true=y_true, y_pred=y_pred)

            auc_dict.update({f'{dset}_tpr': results['tpr'].tolist(), f'{dset}_fpr': results['fpr'].tolist(),
                             f'{dset}_auroc_thresholds': results['auroc_thresholds'].tolist(),
                             f'{dset}_recall': results['recall'].tolist(),
                             f'{dset}_precision': results['precision'].tolist(),
                             f'{dset}_auprc_thresholds': results['auprc_thresholds'].tolist(), "epoch": self.epoch})

            df_dset_results = pd.DataFrame({"epoch": self.epoch, "acc": acc, "loss": loss, "dset": dset,
                                            "auroc": results["auc"],
                                            "auprc": results["auprc"],
                                            # "tn": results["tn"],
                                            # "fp": results['fp'],
                                            # "fn": results['fn'],
                                            # "tp": results['tp']
                                            }, index=[0])

            df_results = pd.concat([df_results, df_dset_results])

        return df_results, auc_dict

    def save_results(self):
        """For each epoch, save results, and model hyperparameters (if not already done).
        Depending on save frequency, save model (and optimizer) weights.
        """
        df_epoch_results, auc_details = self.preprocess_results()
        df_epoch_results["fold"] = self.fold

        if "val" in self.dsets:
            self.train_val_auprc = df_epoch_results["auprc"].tolist()
            self.curr_val_loss = df_epoch_results.loc[df_epoch_results.dset == "val", "loss"].iloc[0]
        elif "train" in self.dsets:
            self.train_val_auprc = df_epoch_results["auprc"].iloc[0], 0

            # Save results
        if os.path.exists(self.results_summary_path):
            df_results_accum = pd.read_csv(self.results_summary_path)
            with open(self.auc_path) as f:
                auc_accum = json.load(f)
        else:
            df_results_accum = pd.DataFrame()
            auc_accum = {}

        df_results_accum = pd.concat([df_results_accum, df_epoch_results])
        df_results_accum.to_csv(self.results_summary_path, index=False)

        auc_accum.update(auc_details)
        with open(self.auc_path, 'w') as outfile:
            json.dump(auc_accum, outfile, sort_keys=True, indent=4)

    def process_results(self, epoch, dsets, y_train, y_val, y_test):
        """Main method of Results."""
        self.epoch = epoch
        self.dsets = dsets
        self.concat_results()
        self.verify_length(y_train, y_val, y_test)
        self.save_results()
