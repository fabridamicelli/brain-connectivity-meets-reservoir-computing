import numpy as np


def make_X_y(
    pattern_length=5,
    low=0.0,
    high=1.0,
    n_trials=5000
):
    """Generate input-output patterns"""
    X, y = generate_input_output_patterns(
        pattern_length=pattern_length,
        low=low,
        high=high,
        n_trials=n_trials
    )
    return X, y

def generate_input_output_patterns(
    pattern_length=5,
    low=0.0,
    high=1.0,
    n_trials=100
):
    """
    Generate pattern to memorize with length N and from a uniform distribution
    between low and high values.
    Trials have a memorize period (the generated numbers=pattern_length)
    and a recall period, that is, 0s=pattern_length. The trials are padded with
    zeros and ones with 1 denoting "recall cue". Thus, trials are 2D arrays.
    """
    all_input_trials = None
    all_output_trials = None

    for tr in range(0, n_trials):

        # Create here standard blocks of the trials, namely the cue and "null input"
        # The cue is a 1 on a channel that is not used for the patterns,
        # so concatanate a vector with 0s when we have a trial with input
        # (the patterns to be memomorized) and a 1 to denote the recall cue
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

    all_input_trials = all_input_trials.T
    all_output_trials = all_output_trials.T

    return all_input_trials, all_output_trials
