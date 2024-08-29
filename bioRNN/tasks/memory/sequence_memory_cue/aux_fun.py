#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import random
import numpy as np

# import torch


def group_shuffle(X, Y, indexes):
    unique_indexes = np.unique(indexes)
    unique_indexes = unique_indexes[np.random.permutation(len(unique_indexes))]

    rearrange_idx = None

    for i in range(0, len(unique_indexes)):
        idx = np.where(unique_indexes[i] == indexes)[0]
        if rearrange_idx is None:
            rearrange_idx = idx
        else:
            rearrange_idx = np.hstack((rearrange_idx, idx))

    X = X[rearrange_idx, :]
    Y = Y[rearrange_idx, :]
    indexes = indexes[rearrange_idx]

    return X, Y, indexes


def combo_params(params):
    # Create a list with tuples denoting all possible combos of values to run
    # the model with. This will be constructed by conjointly taking into
    # account the wrapper_params, trial_params and model paramas dictionaries.
    all_values = []
    all_keys = []

    # assemble the values of the dictionaries in a list
    for value in params.values():
        all_values.append(value)

    # assemble the keys of the dictionaries in a list
    for keys in params.keys():
        all_keys.append(keys)

    all_combos = list(itertools.product(*all_values))

    return all_combos, all_keys


def rand_net(A):
    nodes = A.shape[0]
    values = A[torch.where(A)]
    edges = len(values)

    X = torch.zeros((nodes, nodes)).double()
    idx = torch.where(X == 0)
    rand_idx = random.sample(range(0, len(idx[0])), len(idx[0]))
    X[idx[0][rand_idx[0:edges]], idx[1][rand_idx[0:edges]]] = values

    return X


def map_weights_to_template(w_template=None, w_to_map=None):
    X = torch.zeros(w_template.shape).double()

    idx = torch.where(w_template != 0)
    w_template_values = w_template[idx]
    w_to_map_values = w_to_map[idx]

    (sorted_w_template, sorted_index_w_template) = torch.sort(
        w_template_values, dim=0, descending=True
    )

    (sorted_w_to_map, sorted_index_w_to_map) = torch.sort(
        w_to_map_values, dim=0, descending=True
    )

    X[
        idx[0][sorted_index_w_template], idx[1][sorted_index_w_template]
    ] = sorted_w_to_map

    return X


# Auxiliary function to get the desired parameters from the model
# model: the model from which we should fetch parameters
# params_to_get: a list of str specifying the names of the params to be fetched
def get_model_params(model, params_to_get):
    params_names = []
    params_values = []

    for name, param in zip(model.named_parameters(), model.parameters()):
        if name[0] in params_to_get:
            params_names.append(name[0])
            params_values.append(param.data.clone())

    return params_values, params_names


# Classification accuracy
def calc_accuracy(output=None, labels=None):
    _, predicted = torch.max(output.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    acc = 100 * (correct / total)

    return acc
