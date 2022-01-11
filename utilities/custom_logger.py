import os

import pandas as pd
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.distributed import rank_zero_only


class FriendlyCSVLogger(CSVLogger):
    @rank_zero_only
    def finalize(self, status=None) -> None:
        metrics_path = os.path.join(self.experiment.metrics_file_path)

        def _collapse_epoch(df_):
            """Collapse epoch results to include all non-NA values"""
            row = pd.melt(df_).dropna().drop_duplicates().rename(columns={"variable": "index"})
            return row.set_index("index").T.reset_index(drop=True)

        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df = df.drop(columns=["step"])
            df = df.groupby(by=['epoch'], as_index=False).apply(lambda df_: _collapse_epoch(df_))
            df.to_csv(os.path.join(self.log_dir, "history.csv"), index=False)
