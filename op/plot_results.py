import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure plot settings
plt.style.use("seaborn-colorblind")
matplotlib.rcParams.update({"font.size": 12})

################################################################################
#                                  Constants                                   #
################################################################################
RESULTS_DIR = "C:/Users/Stanley Hua/projects/temporal_hydronephrosis/results"


################################################################################
#                                  Functions                                   #
################################################################################
def grouped_barplot(data, x, y, hue, yerr_low, yerr_high, legend=False,
                    xlabel=None, ylabel=None, ax=None,
                    **plot_kwargs):
    """
    Create grouped bar plot with custom confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data
    x : str
        Name of primary column to group by
    y : str
        Name of column with bar values
    hue : str
        Name of secondary column to group by
    yerr_low : str
        Name of column to subtract y from to create LOWER bound on confidence
        interval
    yerr_high : str
        Name of column to subtract y from to create UPPER bound on confidence
        interval
    legend : bool, optional
        If True, add legend to figure, by default False.
    **plot_kwargs : keyword arguments to pass into `matplotlib.pyplot.bar`

    Returns
    -------
    matplotlib.pyplot.Axis.axis
        Grouped bar plot with custom confidence intervals
    """
    # Get unique values for x and hue
    x_unique = data[x].unique()
    xticks = np.arange(len(x_unique))
    hue_unique = data[hue].unique()

    # Bar-specific constants
    offsets = np.arange(len(hue_unique)) - np.arange(len(hue_unique)).mean()
    offsets /= len(hue_unique) + 1.
    width= np.diff(offsets).mean()

    # Create figure
    if ax is None:
        _, ax = plt.subplots()

    # Create bar plot per hue group
    for i, hue_group in enumerate(hue_unique):
        df_group = data[data[hue] == hue_group]
        ax.bar(
            x=xticks+offsets[i],
            height=df_group[y].values,
            width=width,
            label="{} {}".format(hue, hue_group),
            yerr=abs(df_group[[yerr_low, yerr_high]].T.to_numpy()),
            **plot_kwargs)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set x-axis ticks
    ax.set_xticks(xticks, x_unique)

    if legend:
        ax.legend()

    return ax


def multiplot_separately(df):
    """
    Plot AUROC/AUPRC metrics across models for each hospital separately
    """
    hospitals = ["SickKids", "Stanford", "CHOP"]
    fig, axs = plt.subplots(
        nrows=len(hospitals),
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(12, 7))

    for i, hospital in enumerate(hospitals):
        df_hospital = df[df.Dataset == hospital]

        # Plot AUROC
        axs[i][0].grid(axis="x")
        axs[i][0].bar(
            x=df_hospital["Model"], height=df_hospital["AUROC"],
            color="#377eb8",
            capsize=5,
            alpha=0.85,
            yerr=abs(df_hospital[["AUROC_5_delta", "AUROC_95_delta"]].T.to_numpy()))

        # Plot AUPRC
        axs[i][1].grid(axis="x")
        axs[i][1].bar(
            x=df_hospital["Model"], height=df_hospital["AUPRC"],
            color="#984ea3",
            capsize=5,
            alpha=0.85,
            yerr=abs(df_hospital[["AUPRC_5_delta", "AUPRC_95_delta"]].T.to_numpy()))

    # Configure space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.001)

    # Figure x-axis and y-axis labels
    fig.supxlabel("Model")
    fig.supylabel("Value")

    # Create legend
    metric_colors = {"AUROC":"#377eb8", "AUPRC":"#984ea3"}
    metrics = list(metric_colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=metric_colors[metric]) \
        for metric in metrics]
    fig.legend(
        handles, metrics,
        title="Metric",
        # frameon=True,
        loc=7, bbox_to_anchor=(1, 0.5)
        )
    fig.tight_layout()
    fig.subplots_adjust(right=0.825)

    plt.show()


################################################################################
#                                  Main Flow                                   #
################################################################################
if __name__ == "__main__":
    df = pd.read_excel(RESULTS_DIR + "/final/results_sipaim.xlsx")

    # Calculate delta for CI
    df["AUROC_5_delta"] = df["AUROC_5"] - df["AUROC"]
    df["AUROC_95_delta"] = df["AUROC_95"] - df["AUROC"]
    df["AUPRC_5_delta"] = df["AUPRC_5"] - df["AUPRC"]
    df["AUPRC_95_delta"] = df["AUPRC_95"] - df["AUPRC"]

    # Rename models
    rename_models = {
        "Baseline": "Last US",
        "Avg. Prediction": "Avg. Pred.",
        "Conv. Pooling": "Conv. Pool"
    }
    df["Model"] = df["Model"].map(lambda x: rename_models.get(x, x))

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1,
        sharex=True, sharey=True,
        figsize=(10, 7))

    grouped_barplot(
        data=df,
        x="Dataset",
        y="AUROC",
        hue="Model",
        yerr_low="AUROC_5_delta", yerr_high="AUROC_95_delta",
        alpha=0.8,
        capsize=5,
        ax=ax1)

    grouped_barplot(
        data=df,
        x="Dataset",
        y="AUPRC",
        hue="Model",
        yerr_low="AUPRC_5_delta",
        yerr_high="AUPRC_95_delta",
        alpha=0.8,
        capsize=5,
        ax=ax2)

    # Add grid background
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color="gray", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color="gray", alpha=0.6)

    # Configure space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.001)

    # Figure x-axis and y-axis labels
    fig.supxlabel("Hospital")
    fig.supylabel("Performance")

    # Create legend
    handles, labels = ax1.get_legend_handles_labels()
    labels = [" ".join(label.split(" ")[1:]) for label in labels]
    fig.legend(
        handles, labels,
        title="Model",
        frameon=True,
        loc=7, bbox_to_anchor=(1, 0.5)
        )
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    plt.show()
