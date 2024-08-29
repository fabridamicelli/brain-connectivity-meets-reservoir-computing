"""
Grid of hyperparameters tested for all tasks.
"""
import numpy as np

from echoes.utils import relu

#######################################################################################
# Hyperparameters tuning
#######################################################################################

common_grid = {
    "spectral_radius": np.arange(91, 100, 2) * .01,
    "input_scaling": 10. ** np.arange(-6, 1),
    "leak_rate": [.6, .8, 1.],
    "n_transient": [100],
    "bias": [1],
}

# Sequential memory task
param_grid_memory_sequence = {
    **common_grid,
    **{"activation_out": [relu],}
}

# Memory capacity task
param_grid_memory_capacity = common_grid

# Digits recognition
param_grid_digits = {
    **common_grid,
    **{"activation_out": [relu],}
}
#######################################################################################
# Evaluation with best parameters
#######################################################################################

# Sequential Memory task
best_params_memory_sequence = {
    "n_transient": [100],
    "spectral_radius": [0.99],
    "input_scaling": [1e-5],
    "bias": [1],
    "activation_out": [relu],
}

# Memory capacity task
best_params_memory_capacity = {
    "n_transient": [100],
    "spectral_radius": [0.99],
    "input_scaling": [1e-5],
    "bias": [1],
}
