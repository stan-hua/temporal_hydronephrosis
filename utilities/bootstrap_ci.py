"""
Used to generate 95% confidence intervals via Bias Corrected and accelerated(BCa) bootstrap using arch package.
"""

import json
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
import torch
from arch.bootstrap import IIDBootstrap
from sklearn.utils import resample
from torchmetrics.functional import accuracy, auroc, average_precision

RANDOM_SEED = 42


def bootstrap_ci(preds, labels, n=1000, alpha=0.05, metric_fn=accuracy):
    """DEPRECATED.
    Performs percentile bootstrap on prediction probability and labels. Creates 95% CI on AUROC and AUPRC.
    """
    accum_metric = []

    for i in range(n):
        pred_i, labels_i = resample(preds, labels, random_state=i ** 2)
        metric_i = metric_fn(pred_i, labels_i)
        accum_metric.append(metric_i)

    # Get confidence interval bounds
    left_bound = np.percentile(accum_metric, (alpha / 2) * 100)
    right_bound = np.percentile(accum_metric, (1 - alpha / 2) * 100)

    value = metric_fn(preds, labels)

    # Convert to percentage and round to 2 decimals
    value = round(value.numpy() * 100, 2)
    left_bound = round(left_bound * 100, 2)
    right_bound = round(right_bound * 100, 2)

    return value, (left_bound, right_bound)


def bootstrap_results(results, alpha=0.05, n_bootstrap=9999):
    """Perform bootstrap on model output for all test sets.

    :param results: dictionary where each key is a test set pointing to a dictionary of labels, target and
    model predictions for that test set.
    :param alpha: significance level
    :param n_bootstrap: number of replications
    """
    global TEST_NAMES

    metrics = defaultdict(dict)

    for dset in list(results.keys()):
        if dset == 'uiowa':
            continue

        preds = results[dset]['pred']
        labels = results[dset]['label']

        _auroc = auroc(torch.tensor(preds), torch.tensor(labels))
        _auprc = average_precision(torch.tensor(preds), torch.tensor(labels))

        # Calculate confidence intervals
        def auroc_wrapper(preds_, labels_): return auroc(torch.tensor(preds_), torch.tensor(labels_)).item()
        def auprc_wrapper(preds_, labels_): return average_precision(torch.tensor(preds_), torch.tensor(labels_)).item()
        bootstrap = IIDBootstrap(np.array(preds), np.array(labels))
        bootstrap.seed(RANDOM_SEED)
        auroc_bounds = bootstrap.conf_int(func=auroc_wrapper, reps=n_bootstrap, method='bca', size=1-alpha, tail='two').flatten()
        auprc_bounds = bootstrap.conf_int(func=auprc_wrapper, reps=n_bootstrap, method='bca', size=1-alpha, tail='two').flatten()

        temp_seed = RANDOM_SEED
        while any(np.isnan(auprc_bounds)):
            print("NaN in AUPRC BCa Bootstrap! Repeating calculation...")
            temp_seed += 1
            bootstrap.seed(temp_seed)
            auprc_bounds = bootstrap.conf_int(func=auprc_wrapper, reps=n_bootstrap, method='bca', size=1 - alpha,
                                              tail='two').flatten()

        # Convert to percentages
        _auroc = round(_auroc.numpy() * 100, 2)
        _auprc = round(_auprc.numpy() * 100, 2)
        auroc_bounds = np.round(auroc_bounds * 100, 2)
        auprc_bounds = np.round(auprc_bounds * 100, 2)

        print(f"=={dset}==:")
        print(f"\tAUROC: {_auroc:.2f} [{auroc_bounds[0]:.2f}, {auroc_bounds[1]:.2f}]")
        print(f"\tAUPRC: {_auprc:.2f} [{auprc_bounds[0]:.2f}, {auprc_bounds[1]:.2f}]")
        print()

        # Accumulate metrics
        test_set_name = TEST_NAMES[dset]
        metrics[test_set_name]['AUROC'] = f"{_auroc:.2f} [{auroc_bounds[0]:.2f}, {auroc_bounds[1]:.2f}]"
        metrics[test_set_name]['AUPRC'] = f"{_auprc:.2f} [{auprc_bounds[0]:.2f}, {auprc_bounds[1]:.2f}]"

    return metrics


def save_results(model_names: dict, filename: str):
    """Save results for listed models under filenames.

    :param model_names dictionary mapping of model filename to displayed model name
    :param filename for storing table of experiment results
    """
    global RESULTS_DIR, TEST_NAMES

    df_results = pd.DataFrame()
    for model in model_names:
        with open(f"{RESULTS_DIR}/{model}-test_output-pred.json", "r") as f:
            results = json.load(f)

        metrics = bootstrap_results(results)
        df_metrics = pd.DataFrame(metrics).T.reset_index()
        df_metrics['Model'] = model_names[model]
        df_metrics = df_metrics.rename(columns={'index': "Dataset"})
        df_metrics = df_metrics[["Dataset", "Model", "AUROC", "AUPRC"]]

        df_results = pd.concat([df_results, df_metrics], ignore_index=True)

    df_results['Dataset'] = df_results['Dataset'].astype(
        pd.api.types.CategoricalDtype(categories=list(TEST_NAMES.values())))

    df_results['Model'] = df_results['Model'].astype(
        pd.api.types.CategoricalDtype(categories=list(model_names.values())))

    df_results = df_results.sort_values(by=["Dataset", "Model"])
    df_results.to_csv(f"{RESULTS_DIR}/{filename}.csv", index=False)


def ensemble(model_names: dict):
    """Get outputs of all models specified in <model_names>. Average output of all models, weighing each model equally.
    Then, calculate AUROC and AUPRC with bootstrapped confidence intervals.

    ==Precondition==:
        - model_names is not empty
    """
    accum_results = []
    results = dict()
    ensembled_results = defaultdict(dict)

    for model in model_names:
        with open(f"{RESULTS_DIR}/{model}-test_output-pred.json", "r") as f:
            results = json.load(f)
            accum_results.append(results)

    # Average model output
    for test_set in results.keys():
        if test_set == "uiowa":
            continue

        ensembled_results[test_set]['ids'] = results[test_set]['ids']
        ensembled_results[test_set]['label'] = results[test_set]['label']

        temp_preds = [m[test_set]['pred'] for m in accum_results]
        ensembled_results[test_set]['pred'] = np.array(temp_preds).mean(axis=0)
        print(ensembled_results[test_set]['pred'].shape)

    # Calculate metrics with bootstrap
    results = bootstrap_results(ensembled_results)
    metrics = bootstrap_results(results)
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics['Model'] = "Ensemble"
    df_metrics = df_metrics.rename(columns={'index': "Dataset"})
    df_metrics = df_metrics[["Dataset", "Model", "AUROC", "AUPRC"]]
    print(df_metrics)
    return df_metrics


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

    single_visit_experiment = {"baseline(first_visit)": "First",
                               "baseline": "Latest"}

    multi_visit_experiment = {"baseline": "Baseline", "pretrained-avg_pred": "(Pretrained) Avg. Prediction",
                              "pretrained-conv_pool": "(Pretrained) Conv. Pooling",
                              "pretrained-tsm(last)": "(Pretrained) TSM",
                              "avg_pred": "Avg. Prediction", "conv_pool": "Conv. Pooling", "lstm": "LSTM", "tsm": "TSM"}

    only_multivisit = {"baseline(multi-visit-only)": "Baseline", "avg_pred(multi-visit-only)": "Avg. Prediction",
                       "conv_pool(multi-visit-only)": "Conv. Pooling", "lstm(multi-visit-only)": "LSTM",
                       "tsm(multi-visit-only)": "TSM"}

    # print("Saving baseline experiment...")
    # save_results(single_visit_experiment, "baseline_results(bca)")
    #
    # print("\nSaving multi-visit experiment...")
    # save_results(multi_visit_experiment, "multi-visit_methods_results(bca)")
    #
    # print("\n Saving only multivisit experiment...")
    # save_results(only_multivisit, "only-multivisit_results(bca)")

    df_ensemble = ensemble(multi_visit_experiment)
