"""
Used to generate 95% confidence intervals via bootstrap.
"""

import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.utils import resample
import pandas as pd
from torchmetrics.functional import accuracy, auroc, average_precision


def bootstrap_ci(preds, labels, n=1000, alpha=0.05, metric_fn=accuracy):
    """Performs bootstrap on prediction probability and labels. Creates 95% CI on AUROC and AUPRC.
    """
    accum_metric = []

    for i in range(n):
        pred_i, labels_i = resample(preds, labels, random_state=i ** 2)
        metric_i = metric_fn(pred_i, labels_i)
        accum_metric.append(metric_i)

    # Get confidence interval bounds
    left_bound = np.percentile(accum_metric, (alpha/2) * 100)
    right_bound = np.percentile(accum_metric, (1 - alpha/2) * 100)

    value = metric_fn(preds, labels)

    # Convert to percentage and round to 2 decimals
    value = round(value.numpy() * 100, 2)
    left_bound = round(left_bound * 100, 2)
    right_bound = round(right_bound * 100, 2)

    return value, (left_bound, right_bound)


def bootstrap_results(results):
    """Perform bootstrap on model output for all test test sets.

    :param results dictionary where each key is a test set pointing to a dictionary of labels, target and
    model predictions for that test set.
    """
    global TEST_NAMES

    metrics = defaultdict(dict)

    for dset in list(results.keys()):
        preds = torch.tensor(results[dset]['pred'])
        labels = torch.tensor(results[dset]['label'])
        _auroc, auroc_bounds = bootstrap_ci(preds, labels, metric_fn=auroc)
        _auprc, auprc_bounds = bootstrap_ci(preds, labels, metric_fn=average_precision)

        print(f"=={dset}==:")
        print(f"\tAUROC: {_auroc} [{auroc_bounds[0]}, {auroc_bounds[1]}]")
        print(f"\tAUPRC: {_auprc} [{auprc_bounds[0]}, {auprc_bounds[1]}]")
        print()

        # Accumulate metrics
        test_set_name = TEST_NAMES[dset]
        metrics[test_set_name]['AUROC'] = f"{_auroc} [{auroc_bounds[0]}, {auroc_bounds[1]}]"
        metrics[test_set_name]['AUPRC'] = f"{_auprc} [{auprc_bounds[0]}, {auprc_bounds[1]}]"

    return metrics


if __name__ == '__main__':
    # with open("C:\\Users\\Stanley Hua\\SickKids\\Lauren Erdman - HN_Stanley\\Results/NoFinalLayerFineTuneNoCov_v2_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covFalse_OSFalse.json",
    #           "r") as f:
    #     results = json.load(f)
    # uiowa_results = results['ui']['1']['30']

    TEST_NAMES = {"sk_test": "SickKids Test Set",
                  "st": "SickKids Silent Trial",
                  "stan": "Stanford",
                  "uiowa": "UIowa",
                  "chop": "CHOP"}

    RESULTS_DIR = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/results/final/"

    MODEL_NAMES = {"baseline": "Baseline", "pretrained-avg_pred": "(Pretrained) Avg. Prediction",
                   "pretrained-conv_pool": "(Pretrained) Conv. Pooling", "pretrained-tsm(last)": "(Pretrained) TSM",
                   "avg_pred": "Avg. Prediction", "conv_pool": "Conv. Pooling", "lstm": "LSTM", "tsm": "TSM"}

    MODEL_NAMES = {"baseline(first_visit)": "Baseline (First Visit)",
                   "baseline": "Baseline (Last Visit)"}

    which = "pred"
    from_baseline = False

    df_results = pd.DataFrame()

    for model in MODEL_NAMES:
        with open(f"{RESULTS_DIR}/{model}-test_output-pred.json", "r") as f:
            results = json.load(f)

        metrics = bootstrap_results(results)
        df_metrics = pd.DataFrame(metrics).T.reset_index()
        df_metrics['Model'] = MODEL_NAMES[model]
        df_metrics = df_metrics.rename(columns={'index': "Dataset"})
        df_metrics = df_metrics[["Dataset", "Model", "AUROC", "AUPRC"]]

        df_results = pd.concat([df_results, df_metrics], ignore_index=True)

    df_results = df_results.sort_values(by=["Dataset", "Model"])

    df_results.to_csv(f"{RESULTS_DIR}/results.csv", index=False)
