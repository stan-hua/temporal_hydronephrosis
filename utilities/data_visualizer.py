import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from utilities.dataset_prep import KidneyDataModule

matplotlib.use("Qt5Agg")


class DataViewer:
    """DataViewer class. Used to describe sequences of images, and their class/covariate distribution.
    """

    def __init__(self, img_dict, label_dict, cov_dict, study_ids):
        """
        Initialize DataViewer class.

        :param img_dict: Mapping of patient ID to patient's images
        :param label_dict: Mapping from patient ID to surgery labels
        :param cov_dict: Mapping from patient ID to covariates
        :param study_ids: list of patient IDs
        """
        self.id_side_to_target = {"_".join(u.split("_")[:-1]): v for u, v in label_dict.items()}

        self.img_dict = img_dict
        self.label_dict = label_dict
        self.df_cov = self.process_cov_dict(cov_dict)
        self.df_examples = self.extract_num_visits(self.df_cov)
        self.study_ids = study_ids

    def process_cov_dict(self, cov_dict) -> pd.DataFrame:
        """Extracts identifier related variables from cov_dict. Returns dataframe."""
        df_cov = pd.DataFrame(cov_dict).T.reset_index()
        df_cov['ID_Side'] = df_cov['index'].map(lambda x: "_".join(x.split("_")[:-1]))
        df_cov['ID'] = df_cov['index'].map(lambda x: x.split("_")[0])
        df_cov['Side_L'] = df_cov['index'].map(lambda x: x.split("_")[1])
        df_cov['surgery'] = df_cov['ID_Side'].map(lambda x: self.id_side_to_target[x])
        return df_cov

    def extract_num_visits(self, df_cov):
        """Preprocess dataframe of covariates to get information on each sample (e.g. patient kidney) over multiple
        visits."""

        def _extract_info_per_patient(df_):
            """Get the age range (max - min) of patient across all their visits, and get their number of visits"""
            age_range = max(df_["Age_wks"]) - min(df_['Age_wks'])
            num_visits = len(df_)
            id_ = df_['ID'].iloc[0]
            side = df_['Side_L'].iloc[0]

            row = pd.DataFrame({'age_range': age_range,
                                'num_visits': num_visits,
                                'ID': id_,
                                'Side_L': side}, index=[0])
            return row

        df_cov = df_cov.copy()
        df_examples = df_cov.groupby(by=['ID_Side']).apply(lambda df_: _extract_info_per_patient(df_))
        df_examples = df_examples.reset_index().drop(columns=['level_1'])
        df_examples['surgery'] = df_examples['ID_Side'].map(lambda x: self.id_side_to_target[x])
        return df_examples

    def get_proportion_positive(self):
        """Returns proportion of samples that are positive."""
        return self.df_cov["surgery"].mean()

    def plot_num_visits(self, title=None, ax=None):
        if ax is None:
            fig_, ax = plt.subplots()
        sns.histplot(data=self.df_examples, x='num_visits', hue='surgery', discrete=True, multiple='stack',
                     ax=ax)
        ax.set_xticks(np.arange(min(self.df_examples['num_visits']), max(self.df_examples['num_visits']) + 1, 1))
        ax.set_xlabel("Number of Visits")
        ax.set_ylabel("Count")

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()

    def plot_age(self, title=None):
        fig_, ax = plt.subplots()
        sns.histplot(data=self.df_cov, x='Age_wks', hue='surgery', multiple='stack', ax=ax)
        plt.xlabel("Age (in weeks)")
        plt.ylabel("Count")

        if title is not None:
            plt.title(title)

        plt.tight_layout()
        plt.show()


def plot_umap(embeds, labels, save_dir=None, plot_name="UMAP"):
    """Given intermediate model embeddings, extract and plot 2-dimensional UMAP embeddings and labels for each sample,
    then plot UMAP."""
    # Extract embeddings
    reducer = UMAP(random_state=42)
    umap_embeds = reducer.fit_transform(embeds)

    # Generate plot
    plt.style.use('dark_background')
    sns.scatterplot(x=umap_embeds[:, 0], y=umap_embeds[:, 1],
                    hue=labels,
                    legend="full",
                    alpha=1,
                    palette="tab20",
                    s=7,
                    linewidth=0)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.tight_layout()

    # Save Figure
    if save_dir:
        plt.savefig(f"{save_dir}/{plot_name}.png", bbox_inches='tight', dpi=400)


def load_data():
    from op.model_training import parse_args, modify_args
    args = parse_args()
    modify_args(args)
    args.single_visit = True
    args.test_last_visit = False

    data_params = {'batch_size': 1,
                   'shuffle': True,
                   'num_workers': args.num_workers,
                   'pin_memory': True,
                   'persistent_workers': True if args.num_workers else False}

    dm_ = KidneyDataModule(args, data_params)
    dm_.setup('fit')
    dm_.setup('test')
    dm_.fold = 0
    dm_.train_dataloader()  # to save training set to object
    dm_.val_dataloader()  # to save validation set to object

    return dm_


def describe_data(data_dicts, plot_title=None, ax=None):
    img_dict, label_dict, cov_dict, study_ids = data_dicts

    data_viewer = DataViewer(img_dict, label_dict, cov_dict, study_ids)
    # data_viewer.plot_age(title=plot_title)
    # plt.show()
    data_viewer.plot_num_visits(title=plot_title, ax=ax)

    return data_viewer, data_viewer.df_cov, data_viewer.df_examples


if __name__ == "__main__":
    dm = load_data()

    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    # train_viewer, df_train_cov, df_train_examples = describe_data(dm.train_dicts, plot_title='Ordered Training Set')
    # val_viewer, df_val_cov, df_val_examples = describe_data(dm.val_dicts, plot_title='Ordered Validation Set')
    test_viewer, df_test_cov, df_test_examples = describe_data(dm.test_set, plot_title='SickKids Test Set', ax=axs[0][0])

    st_viewer, df_st_cov, df_st_examples = describe_data(dm.st_test_set, plot_title='SickKids Silent Trial', ax=axs[0][1])
    stan_viewer, df_stan_cov, df_stan_examples = describe_data(dm.stan_test_set, plot_title='Stanford', ax=axs[1][0])
    # ui_viewer, df_ui_cov, df_ui_examples, ui_fig = describe_data(dm.ui_test_set, plot_title='UIowa Data')
    chop_viewer, df_chop_cov, df_chop_examples = describe_data(dm.chop_test_set, plot_title='CHOP', ax=axs[1][1])

    fig.suptitle("Distribution of Repeated Hospital Visits")
    fig.tight_layout()

    handles = axs[1][1].get_legend().legendHandles
    axs[0][0].get_legend().remove()
    axs[0][1].get_legend().remove()
    axs[1][0].get_legend().remove()
    axs[1][1].get_legend().remove()

    fig.subplots_adjust(bottom=0.18)
    fig.legend(handles=handles, labels=["Negative", "Positive"], loc=8, ncol=2)
