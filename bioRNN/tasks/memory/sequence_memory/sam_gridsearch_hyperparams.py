import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from echoes.model_selection import GridSearchTask
from sequence_memory_task import SequenceMemory
from sequence_memory_FD import make_bioRRN
from utils_bioRNN import label_W, AVAILABLE_CONNECTOMES


dir_res = Path(os.environ["DATAICNS"]) / "bioRNN"
n_samples = 15  # number of grid searches


task_params = dict(
    pattern_length = 5,
    n_trials = 200,
#     n_trials = 100,
    high = 1,
    low = -1,
)
n_transient = ((task_params["pattern_length"] * 2) + 1) * 20
param_grid = dict(
    n_inputs=[2],
    n_outputs=[1],
    n_transient=[n_transient],
    spectral_radius=np.arange(90, 101, 2)*.01,
    leak_rate=[.6, .8, 1],
    input_scaling=[.001, .01, .1, .2, .5,.75, 1, 1.25],
#   bias=[0, 1],
#   activation=[relu, np.tanh, sigmoid],
#   fit_only_states=[False, True],
)

for connectome in AVAILABLE_CONNECTOMES:
    results = []
    for _ in range(n_samples):
        W_bio, W_bio_shuffled, W_rnd, W_rnd_k, W_full = make_bioRRN(connectome_name=connectome)
        param_grid["n_reservoir"] = [len(W_bio)]
        param_grid["W"] = [W_bio, W_bio_shuffled, W_rnd, W_rnd_k, W_full]

        result = (GridSearchTask(SequenceMemory, task_params, param_grid, verbose=0, n_jobs=2)
                  .fit()
                  .to_dataframe())

        result["W"] = result.W.apply(
            label_W,
            W_bio=W_bio,
            W_bio_shuffled=W_bio_shuffled,
            W_rnd=W_rnd,
            W_rnd_k=W_rnd_k,
            W_full=W_full,
        )
        results.append(result)

    results = pd.concat(results)
    results["spectral_radius"] = results.spectral_radius.round(decimals=3)
    results.to_csv(dir_res / f"gridsearch_hyperparams/{connectome}.csv", index=False)

    f = sns.catplot(
        x="spectral_radius",
        y="scores",
        data=results,
        hue="W",
        kind="box",
        row="leak_rate",
        col="input_scaling",
        margin_titles=True,
    #     height=4,
    #     aspect=2,
    )
    f.set_xticklabels(rotation=45)
    f.savefig(dir_res / "gridsearch_hyperparams"/ "figs" / f"{connectome}")

    print(f"done with {connectome}")
