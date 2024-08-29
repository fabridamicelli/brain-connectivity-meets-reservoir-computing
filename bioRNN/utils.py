from datetime import datetime as dt
from itertools import product
import os
from pathlib import Path
from shutil import get_terminal_size
from typing import Callable, Mapping, List, Iterable, Union, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score
from tqdm import tqdm

from bio2art import importnet
from utils import clean_df_param_names
from bioRNN.tasks.memory.capacity import forgetting


AVAILABLE_CONNECTOMES = [
    "macaque",
    "drosophila",
    "human",
    "marmoset",
    "mouse-gamanut",
    "mouse-oh",
]

NAMES_MAPPING = {
    "macaque": "Macaque_Normalized",
    "drosophila": "Drosophila",
    "human": "Human_Betzel_Normalized",
    "marmoset": "Marmoset_Normalized",
    "mouse-gamanut": "Mouse_Gamanut_Normalized",
    "mouse-oh": "Mouse_Ypma_Oh",
}

CONNECTOMES_PATH = Path(os.environ["DATAICNS"]) / "connectomes"


def load_connectome(connectome_name: str, path: Path = CONNECTOMES_PATH) -> np.ndarray:
    """Load plain connectome (experimental data)"""
    return np.load(path / f"C_{NAMES_MAPPING[connectome_name]}.npy")


def label_W(W: np.ndarray, W_mapping: Mapping = None) -> str:
    """
    Return label of W according to keys in W_mapping.
    Raise error if not found.
    """
    for name, w in W_mapping.items():
        if (w == W).all():
            if name.startswith("W_"):
                return name[2:]
            return name

    raise ValueError("W not recognized")


def label_W_df(df: pd.DataFrame, W_mapping: Mapping = None) -> pd.DataFrame:
    """Replace W column with the name (label) of the corresponding W"""
    assert "W" in df.columns, "column W is not present in df"
    df["W"] = df.W.apply(label_W, W_mapping=W_mapping)
    return df


def map_rnn2connectome(connectome=None, rnn=None):
    """
    Map weights of rnn onto a template connectome.
    It keeps the rank order of values from the template.
    """
    mapped = np.zeros(connectome.shape)

    idx = np.where(connectome != 0)

    connectome_vals = connectome[idx]
    rnn_vals = rnn[idx]

    sorted_idx = np.argsort(connectome_vals)
    sorted_rnn_vals = np.sort(rnn_vals)

    mapped[idx[0][sorted_idx], idx[1][sorted_idx]] = sorted_rnn_vals

    topology = connectome != 0
    assert (
        spearmanr(mapped[topology], connectome[topology])[0] > 0.9999
    ), "wrong order mapping"
    assert ((mapped != 0) == (connectome != 0)).all(), "wrong topology mapping"

    return mapped


def make_bioRRNs(
    *,
    connectomes_path: str = CONNECTOMES_PATH,
    connectome_name: str,
    k: int = 10,
    neuron_density: Union[int, np.ndarray] = 4,
    target_sparsity: float = 0.8,
    intrinsic_conn: bool = True,
    target_sparsity_intrinsic: float = 1.,
    intrinsic_wei: float = .8,
    rand_partition: bool,
    keep_diag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Wrapper function for bio-reservoir generation.
    Uses bio2art.importnet.from_conn_mat for creating bio-RNN.

    Parameters
    ----------
    k: mean number of connections per node for W_rnd_k, ie
        totally random reservoir.

    Parameters passed to importnet.from_conn_mat function:
        neuron_density
        target_sparsity
        intrinsic_conn
        target_sparsity_intrinsic
        intrinsic_wei
        rand_partition
        keep_diag

    Returns
    -------
    connectomes: bio_rnn and null (random) versions
        W_bio_rank: bioreservoir, ie weigths from a uniform distribution
            mapped onto the connectome template, preserving ranking order of weights.
        W_bio_norank: partially random reservoir, matching template
            topology (nonzeros) but not order (ranking of weights).
        W_rnd_density: random weights matching the density of the connectome.
        W_rnd_k: random weights, unmatched density, ~ k connections per neuron.
        W_full: random weights, (potentially) full connectivity.
    """
    assert (
        connectome_name in NAMES_MAPPING
    ), f"wrong connectome name, choose one of {NAMES_MAPPING.keys()}"

    _, bio_rnn, _ = importnet.from_conn_mat(
        NAMES_MAPPING[connectome_name],
        path_to_connectome_folder=connectomes_path,
        neuron_density=make_neuron_density(connectome_name, neuron_density),
        target_sparsity=target_sparsity,
        intrinsic_conn=intrinsic_conn,
        target_sparsity_intrinsic=target_sparsity_intrinsic,
        intrinsic_wei=intrinsic_wei,
        rand_partition=rand_partition,
        keep_diag=keep_diag
    )
    # Random reservoir initialization
    W = np.random.uniform(low=-1, high=1, size=bio_rnn.shape)

    # Match topology and ranking of weights
    W_bio = map_rnn2connectome(connectome=bio_rnn, rnn=W)

    # Make random controls
    topology = bio_rnn != 0
    W_bio_shuffled = match_topology(W, topology)  # Topology but not ranking
    W_rnd = match_density(W, topology)  # Density of bioRNN to random reservoir
    W_rnd_k = match_k(W, k)  # Unmatched density, pick <k> neighbors per node

    bio_rnns = {
        "W_bio_rank": W_bio,
        "W_bio_norank": W_bio_shuffled,
        "W_rnd_density": W_rnd,
        "W_rnd_k": W_rnd_k,
        "W_rnd_full": W,
    }
    return bio_rnns


def match_topology(W: np.ndarray, topology: np.ndarray) -> np.ndarray:
    """Return copy of W where values are zero where topology == False"""
    W_topology = W.copy()
    W_topology[~topology] = 0
    return W_topology


def match_density(W: np.ndarray, topology: np.ndarray) -> np.ndarray:
    """Return copy of W with only as many links as present in topology"""
    W_density = W.copy()
    n_links = (W != 0).sum()
    n_delete = n_links - topology.sum()
    idx_delete = np.random.choice(n_links, size=n_delete, replace=False)
    new_vals = W[W != 0]
    new_vals[idx_delete] = 0
    W_density[W != 0] = new_vals
    return W_density


def match_k(W: np.ndarray, k: int) -> np.ndarray:
    """Return copy of W with only k links per node"""
    W_k = W.copy()
    nonzero = W != 0
    n_keep = len(W) * k

    new_vals = W[nonzero]
    idx = np.random.permutation(nonzero.sum())
    new_vals[idx[n_keep:]] = 0
    W_k[nonzero] = new_vals
    return W_k


def cv_generator(X, test_size=.2):
    """
    Cross validation splits for gridsearch.
    Since it is a time series, shuffle=False to generate test from last steps.
    """
    yield train_test_split(np.arange(X.shape[0]), shuffle=False, test_size=test_size)


def remove_transient(n_transient=100):
    """
    Decorator to remove initial n_transient values (first dimension).
    Tested for r2_score, mean_squared_error.
    Usage:
    >> r2 = remove_transient(n_transient=200)(r2_score)
    >> r2(y_true, y_pred)

    Alternatively:

    >> @remove_transient(n_transient=200)
    >> def custom_score(y_true, y_pred, **kwargs):
           return compute_something(y_true, y_pred, **kwargs)
    >> custom_score(y_true, y_pred)
    """
    def remover(score_func):
        def score(y_true, y_pred, sample_weight=None, **kwargs):
            y_true, y_pred = y_true[n_transient:], y_pred[n_transient:]
            if sample_weight is not None:
                sample_weight = sample_weight[n_transient:]
            return score_func(y_true, y_pred, sample_weight=sample_weight, **kwargs)
        return score
    return remover


def remove_fixation(score_func):
    """
    Decorate score_func removing where y_true == 0 (fixation time).
    Tested for r2_score and mean_squared_error.
    """
    def score(y_true, y_pred, sample_weight=None, **kwargs):
        mask = y_true != 0
        y_true, y_pred = y_true[mask], y_pred[mask]
        return score_func(y_true, y_pred, sample_weight=None, **kwargs)
    return score


def make_scoring(
    scorers: Mapping[str, Callable],
    greater_is_better: List[bool] = None,
    n_transient: int = 0,
    delete_fixation: bool = True,
) -> Mapping[str, Callable]:
    """
    Return scoring dict ready to use in sklearn gridsearch cv.
    It decorates all functions in scorers, removing transient
    and fixation according to the parameters and turns them
    into valid scorers via sklearn make_score().

    scorers: dict
        Mapping of score_funct_name: score_func.
        Example:
            {"r2": sklearn.metrics.r2_score,
             "mse": sklearn.metrics.mean_squared_error}

    greater_is_better: List[bool], default=None (all True)
        List of either True of False to pass to each scorer.
        If None, greater_is_better=True is passed to all scorers.
    """
    if greater_is_better is None:
        greater_is_better = [True] * len(scorers)
    if delete_fixation:
        scoring = {
            name: make_scorer(
                remove_fixation(
                    remove_transient(n_transient)(func)
                ),
                greater_is_better=better
            )
            for (name, func), better in zip(scorers.items(), greater_is_better)
        }
        return scoring

    scoring = {
        name: make_scorer(
            remove_transient(n_transient)(func),
            greater_is_better=better
        )
        for (name, func), better in zip(scorers.items(), greater_is_better)
    }
    return scoring


def clean_col_names(df: pd.DataFrame):
    df = clean_df_param_names(df)
    df.drop(
        inplace=True,
        columns=[
            c for c in df.columns
            if any((
                c.startswith(("split0", "std_", "rank_")),
                c.endswith("_time"),
                c == "params"
            ))
        ]
    )
    df.rename(
        {"mean_test_r2": "r2",
         "mean_test_mse": "mse"},
        axis="columns",
        inplace=True
    )
    return df


def get_best_params(
    df: pd.DataFrame,
    by: str = "r2",
    params: Iterable = ["input_scaling", "spectral_radius", "leak_rate"],
):
    """
    Return dict {param_name: best_performant_parameter_value}.
    Best (mean) performing gridpoint results of the grid of params evaluated,
    across all pattern lengths.

    params: list-like of parameters (assumed to be column names in df).
    """
    df = (
        df
        .groupby(by=params)
        .mean()
        .sort_values(by=by, ascending=False)
        .reset_index()
    )
    best_params = dict(zip(params, df.iloc[0][params]))
    return best_params


def make_neuron_density(connectome: str, neuron_density: Union[int, np.ndarray]) -> np.ndarray:
    """
    Return vector with same connectome length, filled with neuron_density.
    If neuron_density is a vector, it does nothing to it – just returns it.
    """
    if isinstance(neuron_density, np.ndarray):
        return neuron_density

    n_neurons = {
        "drosophila": 49,
        "human": 57,
        "macaque": 29,
        "marmoset": 55,
        "mouse-gamanut": 19,
        "mouse-oh": 56
    }

    return np.full(n_neurons[connectome], neuron_density, dtype=np.int)


def make_bioRRNs_old(
    connectomes_path: str = CONNECTOMES_PATH,
    connectome_name: str = "macaque",
    k: int = 10,
    neuron_density: Union[int, np.ndarray] = 4,
    target_sparsity: float = 0.8,
    intrinsic_conn: bool = True,
    target_sparsity_intrinsic: float = 1.,
    rand_partition: bool = True,
    keep_diag: bool = True
) -> Dict:
    """
    DEPRECATED

    Wrapper function for bio-reservoir generation.
    Uses bio2art.importnet.from_conn_mat for creating bio-RNN.

    Parameters
    ----------
    k: mean number of connections per node for W_rnd_k, ie
        totally random reservoir.

    Parameters passed to importnet.from_conn_mat function:
        neuron_density
        target_sparsity
        intrinsic_conn
        target_sparsity_intrinsic
        rand_partition
        keep_diag

    Returns
    -------
    bio_rnns: dict
        W_bio: bioreservoir, ie weigths from a uniform distribution
            mapped onto the connectome template, preserving ranking order of weights.
        W_bio_shuffled: partially random reservoir, matching template
            topology (nonzeros) but not order (ranking of weights).
        W_rnd: random weights matching the density of the connectome.
        W_rnd_k: random weights, unmatched density, ~ k connections per neuron.
        W_full: random weights, (potentially) full connectivity.
    """
    assert (
        connectome_name in NAMES_MAPPING
    ), f"wrong connectome name, choose one of {NAMES_MAPPING.keys()}"

    _, bio_rnn, _ = importnet.from_conn_mat(
        NAMES_MAPPING[connectome_name],
        path_to_connectome_folder=connectomes_path,
        neuron_density=make_neuron_density(connectome_name, neuron_density),
        target_sparsity=target_sparsity,
        intrinsic_conn=intrinsic_conn,
        target_sparsity_intrinsic=target_sparsity_intrinsic,
        rand_partition=rand_partition,
        keep_diag=keep_diag
    )
    topology = bio_rnn != 0

    # Random reservoir initialization
    W = np.random.uniform(low=-1, high=1, size=bio_rnn.shape)

    # Match topology and ranking of weights
    W_bio = map_rnn2connectome(connectome=bio_rnn, rnn=W)

    # Make random controls

    # Topology but not ranking
    W_bio_shuffled = W.copy()
    W_bio_shuffled[~topology] = 0

    # Match density of random reservoir to connectome
    W_rnd = W.copy()
    n_links = (W != 0).sum()
    n_delete = n_links - topology.sum()
    idx_delete = np.random.choice(n_links, size=n_delete, replace=False)
    new_vals = W[W != 0]
    new_vals[idx_delete] = 0
    W_rnd[W != 0] = new_vals

    # Unmatched density, pick <k> neighbors per node
    W_rnd_k = W.copy()
    nonzero = W != 0
    n_keep = len(W) * k

    new_vals = W[nonzero]
    idx = np.random.permutation(nonzero.sum())
    new_vals[idx[n_keep:]] = 0
    W_rnd_k[nonzero] = new_vals

    bio_rnns = {
        "W_bio": W_bio,
        "W_bio_shuffled": W_bio_shuffled,
        "W_rnd": W_rnd,
        "W_rnd_k": W_rnd_k,
        "W_full": W,
    }
    return bio_rnns


def concat_results_randpart_pattlen(
    path: Path,
    filename_base: str,
    rand_partitions: Iterable = [True, False],
    pattern_lengths: Union[Iterable, None] = range(5, 31, 5),
) -> pd.DataFrame:
    """
    Collect and concatenate dataframes of experiments with different parameters
    rand_partition and pattern_lengths.
    If pattern_lengths is None, only rand_partitions are concatenated.
    """
    dfs = []
    if pattern_lengths is None:
        for rand_partition in tqdm(rand_partitions, total=len(rand_partitions)):
            filename = path / (filename_base + f"rand-part-{rand_partition}")
            df = pd.read_csv(filename)
            dfs.append(df)

    else:
        for rand_partition, pattern_length in tqdm(product(rand_partitions, pattern_lengths),
                                                   total=len(rand_partitions)*len(pattern_lengths)):
            filename = path / (filename_base + f"rand-part-{rand_partition}_patt-len-{pattern_length}")
            df = pd.read_csv(filename)
            df.insert(len(df.columns) - 2, "pattern_length", [pattern_length] * len(df))
            dfs.append(df)
    return pd.concat(dfs)


def score_without_transient(y_pred, y_true, n_transient=100, score_func=r2_score):
    """
    Score regression prediction coming from arbitrary task and with arbitrary
    score_func.
    The first n_transient steps are eliminated before computing score.
    """
    f = remove_fixation(remove_transient(n_transient=n_transient)(score_func))
    return f(y_pred, y_true)


def make_sklearn_scorer(
    n_transient=100,
    score_func=r2_score,
    delete_fixation=True,
    delete_transient=True,
):
    if not remove_transient and not remove_fixation:
        raise ValueError("Either remove_transient or remove_fixation must be True")
    """Return a sklearn scorer (ready to pass to scoring in GridSearch)"""
    if delete_fixation and delete_transient:
        return make_scorer(remove_fixation(remove_transient(n_transient=n_transient)(score_func)))
    return make_scorer(remove_transient(n_transient=n_transient)(score_func))


def make_r2_scorer(n_transient=None) -> Dict:
    """Make sklearn scorer with R2 for task, eliminating initial transient from evaluation"""
    return {"r2": make_sklearn_scorer(n_transient=n_transient, score_func=r2_score)}


def make_memory_capacity_scorer(n_transient=None) -> Dict:
    def memory_capacity(y_true, y_pred, **kwargs):
        _, mc = forgetting(y_true, y_pred)
        return mc
    scorer = make_scorer(remove_transient(n_transient=n_transient)(memory_capacity))
    return {"memory_capacity": scorer}


def create_fill_cols(df: pd. DataFrame, loc_col_val: List[Tuple[int, str, float]]) -> pd.DataFrame:
    """
    Return copy of df with new columns.
    loc_col_val specifies (location, column, value) to insert using.df.insert.
    It fills the whole column with value in loc_col_val.
    """
    df = df.copy()
    for loc, col, value in loc_col_val:
        df.insert(loc, col, value)
    return df


def print_progress(**params: float) -> None:
    width, _ = get_terminal_size()
    print("".center(width, "="))
    print(dt.now())
    print(", ".join([f"{key}={val}" for key, val in params.items()]).center(width, ".")
    )
    print("".center(width, "="))


def assert_args_not_none(locals_items: Dict) -> None:
    """Check that all arguments are not None"""
    if invalid := [arg for arg, val in locals_items if val is None]:
        raise ValueError(
            f"All arguments must be defined, the following are None: {invalid}"
        )


def capitalize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return dataframe with columns capitalized."""
    for col in columns:
        df[col] = df[col].str.capitalize()
    return df

def refactor_names(
    *,
    df: pd.DataFrame,
    capitalize_cols: List[str] = ["connectome",],
    to_replace: Dict[str, str] = {
        "bio_rank": "Bio (rank)",
        "bio_norank": "Bio (no-rank)",
        "rnd_k": "Random (k)",
        "rnd_density": "Random (density)",
        "rnd_full": "Random (full)",
    },
    to_rename: Dict[str, str] = None,
) -> pd.DataFrame:
    """Capitalize cols in capitalize_cols and replace names in to_replace"""
    if to_replace is not None:
        df = df.replace(to_replace)
    if capitalize_cols is not None:
        df = capitalize_columns(df, columns=capitalize_cols)
    if to_rename is not None:
        df = df.rename(columns=to_rename)
    return df
