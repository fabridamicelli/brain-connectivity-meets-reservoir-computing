from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

from echoes import ESNRegressor
from bioRNN.utils import (
    clean_col_names,
    label_W_df,
    cv_generator,
    make_bioRRN_mapping,
    make_scoring,
)
from bioRNN.memory_tasks.sequence_memory_cue import tasks


def main(
    dir_res=None,
    connectomes_path=None,
    connectomes=None,
    n_jobs=None,
    n_samples=None,
    n_trials_discard=None,
    pattern_length=None,
    task_params=None,
    param_grid=None,
):
    start_time = datetime.now()

    n_transient = int(task_params["pattern_length"] * n_trials_discard)
    scoring = make_scoring(
        {"r2": r2_score, "mse": mean_squared_error},
        greater_is_better=[True, False],
        n_transient=n_transient,
        delete_fixation=True,
    )

    for connectome in connectomes:
        results = []
        for _ in range(n_samples):
            W_mapping = make_bioRRN_mapping(
                connectome_name=connectome, connectomes_path=connectomes_path
            )
            param_grid["W"] = list(W_mapping.values())

            (X_train, X_test), (y_train, y_test), trials_idx = tasks.create_trials(
                task_params
            )
            cv = cv_generator(X_train, test_size=0.2)

            grid = GridSearchCV(
                ESNRegressor(n_transient=n_transient),
                param_grid,
                cv=cv,
                scoring=scoring,
                refit="r2",
                n_jobs=n_jobs,
            ).fit(X_train, y_train)

            result = clean_col_names(pd.DataFrame(grid.cv_results_))
            result = label_W_df(result, W_mapping)
            results.append(result)

        results = pd.concat(results, axis=0)
        results["spectral_radius"] = results.spectral_radius.astype(float).round(
            decimals=2
        )
        results["mse"] = -results.mse
        results.to_csv(
            Path(dir_res) / f"{connectome}_patlen{pattern_length}.csv", index=False
        )

        # Plot grid results
        for metric in ["r2", "mse"]:
            f = sns.catplot(
                x="spectral_radius",
                y=metric,
                data=results,
                hue="W",
                kind="box",
                row="leak_rate",
                col="input_scaling",
                margin_titles=True,
            )
            f.set_xticklabels(rotation=45)
            f.savefig(
                dir_res / "figs" / f"{connectome}_patlen{pattern_length}_{metric}"
            )
            plt.close()

        print(f"{datetime.now()} --> finished {connectome} patlen {pattern_length}")


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "--> started")
    DIR_RES = (
        Path(os.environ["DATAICNS"])
        / "bioRNN/sequence-memory-cue/gridsearch_hyperparams"
    )
    CONNECTOMES_PATH = Path(os.environ["HOME"]) / "Dropbox/code/bioRNN/connectomes"
    CONNECTOMES = ["human", "macaque", "marmoset"]

    for patlen in [5, 10, 15, 20, 25]:
        main(
            dir_res=DIR_RES,
            connectomes_path=CONNECTOMES_PATH,
            connectomes=CONNECTOMES,
            n_jobs=5,
            n_samples=10,  # per gridpoint
            n_trials_discard=20,
            pattern_length=patlen,
            task_params={
                "task_name": "seq_mem",
                "nr_of_trials": 2000,
                "pattern_length": patlen,
                "low": 0.000001,
                "high": 1.0,
                "train_size": 0.75,
            },
            param_grid={
                "spectral_radius": np.arange(80, 101, 5) * 0.01,
                "input_scaling": 1 / 10 ** np.arange(8, -1, -1),
                "leak_rate": [0.8, 0.9, 1],
            },
        )

    print(f"{datetime.now() - start_time}  --> total time")
