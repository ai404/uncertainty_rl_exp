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
You can change some experiment parameters using the global variables in *main.py*:
  * `EXP_NAME`: changes the name of the experiment. It will be the name on the plot as well as the name of the folder in which the results will be saved in the `/experiments` repository.
  * `NB_RUNS`: changes the amount of runs for which the algorithm is trained. More runs mean a more defined average score but also a longer computation time.
  * `NB_ARMS`: it is the amount of arms in the environment class. Yes, it could easily be automatically set. But did not do it yet. Don't forget to set it right :)
  * `ENVIRONMENT`: it is the environment class that you want to use (see Files section for more details)
  * `FIXED_NB_H1`: it is the amount of steps as a function of the problem copmlexity (H1) that you want to go through at each training episode.

In *main.py*, you can change the algorithm you wish to use in the `MainLoop __init__` function:
  * `self.algo = CHOOSEAlgo(self.delta, self.epsilon, self.omega)`


### Files
  * *environment.py* contains two environment classes:
    * `ArticleEnvironment` reproduces the 6 arms bandit described in the paper *Best-arm identification algorithms for multi-armed bandits in the fixed confidence setting*, by Kevin Jamieson and Robert Nowak, CISS, 2014
    * `BookEnvironment` reproduces the 10 arms bandit testbed described in the Reinforcement Learning book by Sutton and Barto.
  * *algorithms.py* contains three algorithm classes, taking as arguments *epsilon*, *delta* the confidence, and *omega* the set of possible arms:, 
    * `ActionEliminationAlgo` reproduces the Action Elimination algorithm as described in the paper.
    * `UCBAlgo` reproduces the UCB algorithm as described in the paper.
    * `LUCBAlgo` reproduces the LUCB algorithm as described in the paper.
  * *utils.py* contains some utility functions used by the algorithms or the plotting program, including the `UFunction` and `CFunction` as defined in the paper.
  * *main.py* is the main file running the whole code. It includes two classes:
    * `MainLoop` is where one episode is run. You can change the end condition (number of steps, or solved condition achieved) in the `findBestArm` function.
    * `Drawer` is where the result plots are created and saved. It could also save csv files - but this has not been implemented yet.
    
