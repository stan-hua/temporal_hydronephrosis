"""
Used to generate 95% confidence intervals via bootstrap.
"""

import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.utils import resample
from torchmetrics.functional import accuracy, auroc, average_precision

TEST_NAMES = {"sk_test": "SickKids Test Set",
              "st": "SickKids Silent Trial",
              "stan": "Stanford",
              "uiowa": "UIowa",
              "chop": "CHOP"}


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
        metrics[dset]['AUROC'] = f"{_auroc} [{auroc_bounds[0]}, {auroc_bounds[1]}]"
        metrics[dset]['AUPRC'] = f"{_auprc} [{auprc_bounds[0]}, {auprc_bounds[1]}]"

    return metrics


if __name__ == '__main__':
    # with open("C:\\Users\\Stanley Hua\\SickKids\\Lauren Erdman - HN_Stanley\\Results/NoFinalLayerFineTuneNoCov_v2_TrainOnly_40epochs_bs16_lr0.001_RCFalse_covFalse_OSFalse.json",
    #           "r") as f:
    #     results = json.load(f)
    # uiowa_results = results['ui']['1']['30']
    which = ["embed", "pred"][1]
    results_dir = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/results/"
    if which == 'embed':
        curr_results_dir = f"{results_dir}Pretrained_Siamese_Baseline_2022-02-11_22-34-18/"
        filename = f"{curr_results_dir}/baseline-test_output-{which}.json"
    else:
        # curr_results_dir = f"{results_dir}Pretrained_Siamese_Baseline_2022-02-10_23-29-33/"
        # filename = f"{curr_results_dir}/baseline-test_output.json"

        # curr_results_dir = f"{results_dir}Pretrained_Siamese_Baseline_2022-02-14_10-19-32(multi-visit)"
        # filename = f"{curr_results_dir}/baseline-test_output-{which}.json"

        # curr_results_dir = f"{results_dir}Pretrained_Siamese_AvgPred_2022-02-14_10-28-30(multi-visit)"
        # filename = f"{curr_results_dir}/avg_pred-test_output-{which}.json"

        curr_results_dir = f"{results_dir}Pretrained_Siamese_ConvPooling_2022-02-14_10-32-00(multi-visit)"
        filename = f"{curr_results_dir}/conv_pool-test_output-{which}.json"

    with open(filename, "r") as f:
        results = json.load(f)

    if which == "embed":
        embeds = np.array(results['sk_test']['embed'])
        labels = np.array(results['sk_test']['label'])
        ids = results['sk_test']['ids']

        reducer = UMAP(random_state=42)
        umap_embeds = reducer.fit_transform(embeds)

        df = pd.DataFrame(umap_embeds)
        df['label'] = labels
        df['ids'] = ids
        df.loc[(df[0] > 7.27) & (df.label == 0)].ids.map(lambda x: "_".join(x.split("_")[:2])).tolist()
    else:
        metrics = bootstrap_results(results)

