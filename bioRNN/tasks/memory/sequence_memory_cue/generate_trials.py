import tasks

# Choose the task
task_name = "bin_mem"

# Parameters for the trials
# pic_mem
# trial_params={
#             'nr_of_trials': 1000,
#             'trial_length': 5,
#             'trial_matching': True,
#             'rescale':True,
#             'train_size': 0.8,
#             }

# bin_mem
trial_params = {
    "nr_of_trials": 500,
    "trial_length": 5,
    "trial_matching": True,
    "train_size": 0.8,
}

# seq_mem
# trial_params = {
#                 'nr_of_trials': 1000,
#                 'low': 0.,
#                 'high': 1.,
#                 'train_size': 0.8,
#                 }

trial_params["task_name"] = task_name

# Tasks specific params depedning on task choice
trial_params["pattern_length"] = [5]
trial_params["n_back"] = [2]

# Generate trials
X, Y, trials_idx = tasks.create_trials(trial_params)
