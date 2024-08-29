#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from keras.datasets import mnist
# import torch
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit

import aux_fun

# Task functions


# Generate pattern to memorize with length N and from a uniform distribution
# between low and high values.
# Trials have a memorize period (the generated numbers=pattern_length)
# and a recall period, that is, 0s=pattern_length. The trials are padded with
# zeros and ones with 1 denoting "recall cue". Thus, trials are 2D arrays.
def generate_sequence_patterns(pattern_length=3, low=0.0, high=1.0, nr_of_trials=100):
    all_input_trials = None
    all_output_trials = None

    all_trials_index = None

    # import numpy as np

    for tr in range(0, nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input
        # (the patterns to be memorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns.

        # 1 is presented only once, with zeros following it for the "null input"

        null_input = np.zeros((2, pattern_length + 1))

        # Assign the cue at the upper left corner so that the first column of the
        # null input is actually the recall cue.
        null_input[0, 0] = 1

        padding_for_trial = np.zeros((pattern_length,))

        # Generate one trial based on the specifications
        trial = np.random.uniform(low, high, pattern_length)

        # Add the padding that corresponds to a cue=0 (that means no replaying yet,
        # but leanrning the input patterns)
        trial = np.vstack((padding_for_trial, trial))

        input_trial = np.hstack((trial, null_input))

        # Now we can construct the desired ouput. This is basically a "mirrored"
        # version of the input, so construct accordingly: where null_input put
        # the current trial and vice versa.

        # We need no padding for the output (no "cue needed"). Just require 0s
        # when the pattern is being learned.
        null_output = np.zeros(
            (1, pattern_length + 1)
        )  # Add 1 column to have the same length with input

        trial = trial[1:, :]

        output_trial = np.hstack((null_output, trial))

        # Concatanate the generated input/output trials to the the overall
        # trials array
        if all_input_trials is None:
            all_input_trials = input_trial
            all_output_trials = output_trial

        else:
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))

        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1])
        current_index[:] = tr

        if all_trials_index is None:
            all_trials_index = current_index

        else:
            all_trials_index = np.hstack((all_trials_index, current_index))

    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T

    all_trials_index = all_trials_index.T

    return all_input_trials, all_output_trials, all_trials_index


def generate_pic_wm_trials(
    images=None,
    trial_length=5,
    nr_of_trials=100,
    n_back=None,
    trial_matching=False,
    rescale=True,
):
    if ((trial_length - n_back) <= 0) and (n_back is not None):
        raise ValueError("N-Back value must be less than the trial length")

    all_input_trials = None
    all_output_trials = None

    all_trials_index = None

    img_pixels = images.shape[1] * images.shape[2]

    for tr in range(0, nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns.

        # 1 is presented only once, with zeros following it for the "null input"
        null_input = np.zeros((img_pixels + 1, 2))

        # Assign the cue at the upper left corner so that the first column of the
        # null input is actually the recall cue.
        null_input[0, 0] = 1

        padding_for_trial = np.zeros((trial_length,))

        # Generate one trial based on the specifications
        # The last pic in the trial must be also the pic n_back steps
        target_pic_idx = random.randrange(images.shape[0])
        trial = np.zeros((img_pixels, trial_length))

        # Mark the positions of the target picture with 1s and assign the
        # target pic in the correct indexes indicated by len(trial_idxs) and
        # n_back. Take into account if the trials are
        # trial_matching = True or False
        trial_idxs = np.zeros((trial_length,))

        if trial_matching is True:
            trial_idxs[len(trial_idxs) - 1] = 1
            trial_idxs[(len(trial_idxs) - 1) - n_back] = 1
            trial[:, len(trial_idxs) - 1] = images[target_pic_idx, :, :].reshape(
                img_pixels
            )
            trial[:, (len(trial_idxs) - 1) - n_back] = images[
                target_pic_idx, :, :
            ].reshape(img_pixels)
        else:
            trial_idxs[len(trial_idxs) - 1] = 1
            trial[:, len(trial_idxs) - 1] = images[target_pic_idx, :, :].reshape(
                img_pixels
            )

        random_pic_idx = random.sample(
            range(0, images.shape[0] - 1), len(np.where(trial_idxs == 0)[0])
        )

        rand_pic_idxs = np.where(trial_idxs == 0)[0]

        for i in range(0, len(random_pic_idx)):
            trial[:, rand_pic_idxs[i]] = images[random_pic_idx[i], :, :].reshape(
                img_pixels
            )

        # Rescale to 255 if rescale True
        if rescale is True:
            trial = trial / 255

        # Add the padding that corresponds to a cue=0
        # (that means no replaying yet, but learning the input patterns)
        trial = np.vstack((padding_for_trial, trial))

        input_trial = np.hstack((trial, null_input))

        # Now we can construct the desired ouput.
        # What we need is an array with input_trial shape with 3 discrete
        # values:
        # 0:fixation
        # 1:n_back matches=False
        # 2:n_back matches=True
        # Hence, the output trials correspond to the correct labels of
        # a 3-class classification problem
        output_trial = np.zeros(
            (1, trial_length + 2)
        )  # Add 1 column to have the same length with input

        # Assign the correct labeling

        if trial_matching is True:
            output_trial[0, (trial_length + 1) :] = 2
        else:
            output_trial[0, (trial_length + 1) :] = 1

        # Concatanate the generated input/output trials to the overall
        # trials array
        if all_input_trials is None:
            all_input_trials = input_trial
            all_output_trials = output_trial

        else:
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))

        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1])
        current_index[:] = tr

        if all_trials_index is None:
            all_trials_index = current_index

        else:
            all_trials_index = np.hstack((all_trials_index, current_index))

    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T

    all_trials_index = all_trials_index.T

    return all_input_trials, all_output_trials, all_trials_index


def generate_bin_wm_trials(
    trial_length=5, nr_of_trials=100, n_back=None, trial_matching=False
):
    if ((trial_length - n_back) <= 0) and (n_back is not None):
        raise ValueError("N-Back value must be less than the trial length")

    all_input_trials = None
    all_output_trials = None

    all_trials_index = None

    for tr in range(0, nr_of_trials):
        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input
        # (the patterns to be memomorized) and a 1 to denote the recall cue
        # when the reservoir has to replay the patterns.

        # 1 is presented only once, with zeros following it for the "null input"
        null_input = np.zeros((2, 1))

        # Assign the cue at the upper left corner so that the first column of the
        # null input is actually the recall cue.
        # null_input[0,0] = 1

        padding_for_trial = np.zeros((trial_length,))
        padding_for_trial[-1] = 1.0

        # Generate one trial based on the specifications
        # The last pic in the trial must be also the pic n_back steps
        # target_pic_idx = random.randrange(x_train.shape[0])

        # trial = np.zeros((1, trial_length))
        trial = np.random.uniform(0.0, 1.0, trial_length)
        trial = np.reshape(trial, (1, trial_length))

        # Mark the positions of the target picture with 1s and assign the
        # target pic in the correct indexes indicated by len(trial_idxs) and
        # n_back. Take into account if the trials are
        # trial_matching = True or False
        # trial_idxs = np.zeros((trial_length,))

        # target_value = random.sample(range(0, 2), 1)
        target_value = np.random.uniform(0.0, 1.0, 1)

        if trial_matching is True:
            # trial_idxs[len(trial_idxs)-1] = 1
            # trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[0, trial_length - 1] = target_value[0]
            trial[0, (trial_length - n_back - 1)] = target_value[0]
        else:
            # trial_idxs[len(trial_idxs)-1] = 1
            # trial_idxs[(len(trial_idxs)-1) - n_back] = 1
            trial[0, trial_length - 1] = target_value[0]

            # Put the wrong value in n_back position since the trial is False
            # if target_value[0] == 0:
            #     trial[0, len(trial_idxs) - n_back] = 1
            # else:
            #     trial[0, len(trial_idxs) - n_back] = 0

        # rand_trials_idxs = np.where(trial_idxs == 0)[0]
        # random_values = [random.randint(0, 1) for x in range(0, len(rand_trials_idxs))]
        # random_values = np.random.uniform(0., 1., len(rand_trials_idxs))

        # for i in range (0, len(rand_trials_idxs)):
        #    trial[:,rand_trials_idxs[i]] = random_values[i]

        # Add the padding that corresponds to a cue=0
        # (that means no replaying yet, but learning the input patterns)
        trial = np.vstack((padding_for_trial, trial))

        input_trial = np.hstack((trial, null_input))

        # Now we can construct the desired ouput.
        # What we need is an array with input_trial shape with 3 discrete
        # values:
        # 0:fixation
        # 1:n_back matches=False
        # 2:n_back matches=True
        # Hence, the output trials correspond to the correct labels of
        # a 3-class classification problem
        output_trial = np.zeros(
            (1, trial_length + 1)
        )  # Add 1 column to have the same length with input

        # Assign the correct labeling

        if trial_matching is True:
            output_trial[0, trial_length] = 2
        else:
            output_trial[0, trial_length] = 1

        # Concatanate the generated input/output trials to the the overall
        # trials array
        if all_input_trials is None:
            all_input_trials = input_trial
            all_output_trials = output_trial

        else:
            all_input_trials = np.hstack((all_input_trials, input_trial))
            all_output_trials = np.hstack((all_output_trials, output_trial))

        # Construct the indexes to keep track of the trials
        current_index = np.zeros(input_trial.shape[1])
        current_index[:] = tr

        if all_trials_index is None:
            all_trials_index = current_index

        else:
            all_trials_index = np.hstack((all_trials_index, current_index))

    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T

    all_trials_index = all_trials_index.T

    return all_input_trials, all_output_trials, all_trials_index


def wrapper_trials(func):
    def internal(**kwargs):
        print("Generating trials with params:")
        for key in kwargs:
            if key != "images":
                print("%s: %s" % (key, kwargs[key]))

        # Call the function with 'trial_matching':True
        kwargs.update({"trial_matching": True})
        X_match, Y_match, indexes_match = func(**kwargs)

        # Call the function with 'trial_matching':False
        kwargs.update({"trial_matching": False})
        X_non_match, Y_non_match, indexes_non_match = func(**kwargs)

        print("Generating trials with params:")
        for key in kwargs:
            if key != "images":
                print("%s: %s" % (key, kwargs[key]))

        # Increase the indexes_non_match in such a way so that they are
        # a continuation of the indexes_match
        indexes_non_match += np.max(indexes_match) + 1

        # Concatanate all true and false trials
        X = np.vstack((X_match, X_non_match))
        Y = np.vstack((Y_match, Y_non_match))
        indexes = np.hstack((indexes_match, indexes_non_match))

        return X, Y, indexes

    return internal


def create_train_test_trials(X=None, Y=None, indexes=None, train_size=0.2):
    # Create train and validate tests by ensuring that only complete
    # trials are shuffled!
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size)

    for train_idx, validate_idx in gss.split(X, Y, indexes):
        X_train = X[train_idx]
        Y_train = Y[train_idx]

        X_validate = X[validate_idx]
        Y_validate = Y[validate_idx]

    # Seperate idx for train and validate sets
    train_trial_idxs = indexes[train_idx]
    validate_trial_idxs = indexes[validate_idx]

    # Wrap up results in lists. Each position corresponds to train
    # validate sets
    X = [X_train, X_validate]
    Y = [Y_train, Y_validate]
    indexes = [train_trial_idxs, validate_trial_idxs]

    return X, Y, indexes


# Trim zeros in-between experimental periods
def trim_zeros_from_trials(actual=None, predicted=None):
    ind = torch.where(actual != 0)[0]  # use the actual trials for tracking non 0s
    actual_trimmed = actual[ind]
    predicted_trimmed = predicted[ind]

    return actual_trimmed, predicted_trimmed


def create_trials(trial_params):
    if trial_params["task_name"] == "seq_mem":
        pattern_length = trial_params["pattern_length"]
        low = trial_params["low"]
        high = trial_params["high"]
        nr_of_trials = trial_params["nr_of_trials"]

        train_size = trial_params["train_size"]

        # Train and validation test
        (X, Y, indexes) = generate_sequence_patterns(
            pattern_length=pattern_length, low=low, high=high, nr_of_trials=nr_of_trials
        )

    if trial_params["task_name"] == "pic_mem":
        # Load the data
        # Use only the test set - it has
        # less but sufficient samples than the train
        _, (x_test, y_test) = mnist.load_data()

        nr_of_trials = trial_params["nr_of_trials"]
        trial_length = trial_params["trial_length"]
        n_back = trial_params["n_back"]
        trial_matching = trial_params["trial_matching"]
        rescale = trial_params["rescale"]

        train_size = trial_params["train_size"]

        generate_pic_wm_trials_boosted = wrapper_trials(generate_pic_wm_trials)

        X, Y, indexes = generate_pic_wm_trials_boosted(
            images=x_test,
            trial_length=trial_length,
            nr_of_trials=nr_of_trials,
            n_back=n_back,
            trial_matching=trial_matching,
            rescale=rescale,
        )

    if trial_params["task_name"] == "bin_mem":
        nr_of_trials = trial_params["nr_of_trials"]
        trial_length = trial_params["trial_length"]
        n_back = trial_params["n_back"]
        trial_matching = trial_params["trial_matching"]

        train_size = trial_params["train_size"]

        generate_bin_wm_trials_boosted = wrapper_trials(generate_bin_wm_trials)

        X, Y, indexes = generate_bin_wm_trials_boosted(
            trial_length=trial_length,
            nr_of_trials=nr_of_trials,
            n_back=n_back,
            trial_matching=trial_matching,
        )

    # Create train and validate tests by ensuring that only complete trials
    # are
    X, Y, indexes = create_train_test_trials(
        X=X, Y=Y, indexes=indexes, train_size=train_size
    )

    X[0], Y[0], indexes[0] = aux_fun.group_shuffle(X[0], Y[0], indexes[0])
    X[1], Y[1], indexes[1] = aux_fun.group_shuffle(X[1], Y[1], indexes[1])

    return X, Y, indexes
