# Markov Decision Process

## Current software
### Required
Python 3
matplotlib
csv
os
numpy
math
random
collections
time
gym
multiprocessing
tqdm

### How to run
To run a single experiment, in a terminal window, from `.../uncertainty_rl_exp/MDP`, run:

```python main.py```

The results will be saved in the `/experiments` repository.

You can also run several experiments using multiprocessing optimisation with:

```python main_bulk_runner.py```

This will automatically run the experiments with environment parameters described in the *experiments.csv* file.

### How to change the parameters
When running a single experiment, you can change some parameters using the variables in the ` if __name__ == "__main__":` part of *main.py*:
  * `exp_name`: changes the name of the experiment. It will be the name on the plot as well as the name of the folder in which the results will be saved in the `/experiments` repository.
  * `nb_runs`: changes the amount of runs of which the algorithm is trained (results are averaged on the runs).
  * `nb_episodes`: changes the amount of episodes per run for which the algorithm is trained. 
  * `rew_paramsX`: reward parameters of an environment. `rvar_mean_ter` is the mean of the variance of the reward given for reaching the goal state. `rvar_var_ter` is the variance of the variance of the goal reward. `rvar_mean_step` and `rvar_var_ter` are respectively the same parameters for the reward given at each step.
  * `envX`: allows you to choose the environment you want to test your algorithm on. Example: `SparseTabularEnvironment(5, 5, rew_params1)` will create a 5x5 sparse tabular environment. Note: you can create several environments `env1`, `env2`, ..!
  * `algo_paramsX`: parameters of the algorithm you want to use. `action_space` is the action space defined in your environment. The others are specific to your algorithm.
  * `algoX`: allows you to choose the algorithm you want to test. Example: `Sarsa(algo_params1)`, `ModifiedSarsa(algo_params2)`, or `ModifiedMonteCarlo(algo_params3)`. Note: you can create several algorithm instances!

Don't forget to update the `drawer,params = compare([algo1, algo2], [env1, env2], exp_name, nb_runs, nb_episodes)` function with the right environments and algo lists.

The same parameters can be changed in *main_bulk_runner.py* even though they are less well organized.

Other parameters:
You can change the rewards given for reaching the goal state or for taking a step in the `brewards` class of *environments.py*.
The maximal number of steps before closing the episode is a default argument of the tabular environments.

### Files
  * *environment.py* contains several environment classes, all tabular grid worlds with different reward production.
    * `Sparse` only gives a reward at the very end of the episode, be it when the agent reaches the goal state or if it maxes out the number of steps.
    * `Semisparse` gives a reward for each step plus a reward when the agent reaches the goal state.
    * `Dense` is like the semisparse but adds a random reward at each step if it is the first time the agent passes by this state.
  * *algorithms.py* contains the algorithm classes, that are tabular, softmax exploring algorithms:
    * `Sarsa` and its inverse-variance weighted average equivalent `ModifiedSarsa`
    * `MonteCarlo` and its inverse-variance weighted average equivalent `ModifiedMonteCarlo`

  * *utils.py* contains some utility functions used by the algorithms or the plotting program.
  * *main.py* is the main file running the whole code.
  * *main_bulk_runner.py* allows use multiprocessing to run several experiments simultaneaously. These experiments are defined in *experiments.csv*
  * *experiments.csv* contains the details of the environment for the experiments run by *main_bulk_runner.py*
