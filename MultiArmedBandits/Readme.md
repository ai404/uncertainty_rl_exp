# MultiArmed bandit

## Current software
### Required
Python 3
matplotlib
csv
os
numpy
math
random

### How to run
In a terminal window, from `.../uncertainty_rl_exp/MultiArmedBandits`, run:

```python main.py```

The results will be saved in the `/experiments` repository.

### How to change the parameters
You can change some experiment parameters using the variables in the ` if __name__ == "__main__":` part of *main.py*:
  * `EXP_NAME`: changes the name of the experiment. It will be the name on the plot as well as the name of the folder in which the results will be saved in the `/experiments` repository.
  * `NB_RUNS`: changes the amount of runs for which the algorithm is trained. More runs mean a more defined average score but also a longer computation time.
  * `TASK`: it is the task you want to achieve (`AlgoEnvCompare` compares different algos on different environments based on regret, best move probability and mean square estimated error)

Then, you must change the parameters in the task definitions.
For `AlgoEnvCompare`:

  * `NB_STEPS` : number of time steps you want to test your algorithm on.
  * `ENV1`: choose one of the environments you want to test your algorithm on. Example: `UncertainRewardEnvironment()`, or `CertainRewardEnvironment()`.
  * `ENV2`: you can put several enviroments, as many as you want!
  * `EPSILON`: this is the initial value of epsilon in if you use an epsilon-greedy algo.
  * `DECAY`: the epsilon decay factor of your epsilon-greedy algo
  * `ALGO1`: Choose one of the algorithms you want to test. Example: `EpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())` or `ModifiedEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())`
  * `ALGO2`: you can test several algorithms, as many as you want!
  * `DO_LIST`: allows you to choose which combination of algo/environment you want to test. For example: `[[1, 1], [0, 1]]` will run `ALGO1` on `ENV1` and `ENV2` but `ALGO2` only on `ENV2`.

Don't forget to update the `compareAlgosEnv(NB_RUNS, EXP_NAME, [ENV1, ENV2, ENV3], [ALGO0, ALGO1, ALGO2], NB_STEPS, DO_LIST)` function with the right environments and algo lists.



### Files
  * *environment.py* contains several environment classes, with different characteristics.
    * `CertainRewardEnvironment` and its family hands out the reward with a fix variance (it is not perfectly named, I know)
    * `UncertainRewardEnvironment` and its family hands out the reward with a changing variance (not perfeclty named neither)
  * *algorithms.py* contains the algorithm classes, for now epsilon-greedy:
    * `EpsilonGreedyAlgo` is a simple epsilon greedy updating the expected reward of the arms with an unweighted average.
    * `ModifiedEpsilonGreedyAlgo` is the same BUT updates the expected reward using an inverse-variance weighted average.
    * `CheatingEpsilonGreedyAlgo` always knows the best action when greedy - but still expores with probability epsilon.
  * *utils.py* contains some utility functions used by the algorithms or the plotting program.
  * *main.py* is the main file running the whole code. It includes two classes:
    * `MainLoop` is where one episode is run.   
    * `Drawer` is where the result plots and the corresponding csv files are created and saved.
  * *drawdistrib.py* is a helper file that can be used to sample probability distributions and draw them.
