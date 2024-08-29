import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from sequence_memory_FD import make_bioRRN, AVAILABLE_CONNECTOMES
from sequence_memory_task import SequenceMemory
from utils_bioRNN import label_W

dir_res = Path(os.environ["DATAICNS"]) / "bioANN"
n_samples = 10  # samples per constellation

pattern_lengths = np.arange(4, 26)

results = {
    "connectome": [],
    "W": [],
    "pattern_length": [],
    "score": [],
}

esn_params = dict(
    n_inputs=2,
    n_outputs=1,
    spectral_radius=.99,
    n_transient=220,
    input_scaling=0.0001,
)

for connectome in AVAILABLE_CONNECTOMES:
    W_bio, W_bio_shuffled, W_rnd, W_rnd_k, W_full = make_bioRRN(connectome_name=connectome, k=10)
    Ws = (W_bio, W_bio_shuffled, W_rnd, W_rnd_k, W_full)
    W_names = ("W_bio", "W_bio_shuffled", "W_rnd", "W_rnd_k", "W_full")

    for W in Ws:
        esn_params["W"] = W
        esn_params["n_reservoir"] = len(W)

        for pattern_length in pattern_lengths:
            for _ in range(n_samples):
                score = SequenceMemory(
                    n_trials=200, pattern_length=pattern_length, esn_params=esn_params,
                ).score()

                results["pattern_length"].append(pattern_length)
                results["score"].append(score)
                results["connectome"].append(connectome)
                results["W"].append(label_W(W, **dict(zip(W_names, Ws))))

    print(f"done with {connectome}")

results = pd.DataFrame(results)
results.to_csv(dir_res / "pattern_length" / "pattern_length.csv", index=False)

lines = sns.relplot(
    x="pattern_length", y="score", data=results,
    col="connectome", hue="W",
    kind="line",
    col_wrap=3,
)

boxes = sns.catplot(
    x="pattern_length", y="score", data=results,
    col="connectome", hue="W",
    kind="box",
    col_wrap=3,
)

lines.set_xticklabels(rotation=45)
boxes.set_xticklabels(rotation=45)
lines.savefig(dir_res / "pattern_length"/ "figs" / "lines")
boxes.savefig(dir_res / "pattern_length"/ "figs" / "boxes")
