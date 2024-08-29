import numpy as np
from sklearn.metrics import r2_score


def generate_input_output_patterns(pattern_length=5, low=0.0, high=1.0, n_trials=100):
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


def evaluate_performance(predicted, actual, discard=0, low=0.0, high=1.0):
    actual = actual[discard:]
    predicted = predicted[discard:]

    # Use the fact that the positions of interest are the non 0s in
    # the actual array. This is task specific so if the the format
    # of the trials changes then the way to fetch elements also must change!!
    indexes_not_zeros = np.where(actual != 0)[0]
    predicted = predicted[indexes_not_zeros]
    actual = actual[indexes_not_zeros]
    err = mean_squared_error(actual, predicted)
    #     print("r2:", r2_score(actual, predicted))
    # Generate a sequence from the same distribution used for the trials
    # This will function as a "null" baseline
    predicted_rand = np.random.uniform(low, high, len(predicted))
    err_null_random = mean_squared_error(actual, predicted_rand)
    return err, err_null_random, actual, predicted, predicted_rand


def score_task(predicted, actual, discard=0):
    actual = actual[discard:]
    predicted = predicted[discard:]

    # Use the fact that the positions of interest are the non 0s in
    # the actual array. This is task specific so if the the format
    # of the trials changes then the way to fetch elements also must change!!
    indexes_not_zeros = np.where(actual != 0)[0]
    predicted = predicted[indexes_not_zeros]
    actual = actual[indexes_not_zeros]
    return r2_score(actual, predicted)


class SequenceMemory:
    def __init__(
        self, pattern_length=5, low=0.0, high=1.0, n_trials=5000, esn_params=None
    ):
        self.pattern_length = pattern_length
        self.low = low
        self.high = high
        self.n_trials = n_trials
        self.esn_params = esn_params

    def make_data(self):
        """
        Data maker is the same for train and test.
        """
        X, y = generate_input_output_patterns(
            pattern_length=self.pattern_length,
            low=self.low,
            high=self.high,
            n_trials=self.n_trials,
        )
        return X, y

    def fit(self):
        # Do whatever you want, eg train en ESN to remember patterns.
        # For example, something that looks like this
        self.esn = ESNPredictive(**self.esn_params)
        X_train, y_train = self.make_data()
        self.esn.fit(X_train, y_train)
        return self

    def score(self):
        self.fit()
        X_test, y_test = self.make_data()
        y_predicted = self.esn.predict(X_test)
        return score_task(y_predicted, y_test, discard=self.esn.n_transient)
