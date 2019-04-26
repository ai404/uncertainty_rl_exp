from environment import *
from algorithms import *
from utils import *
import time


class Trainer(object):
# Utility class allowing to run an algorithm on an environment.
    def __init__(self, env, algo):
        self.env = env
        self.algo = algo

    def trainOneEpisode(self, render = False):
    # Trains one episode and returns the episode return.
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, reward_noise, rvar, done = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, reward_noise, rvar, next_state, next_action, done])
            state = next_state
            action = next_action
            if render: 
                env.render()
        return self.algo.getReturn()

    def evalAvgReturn(self, numbers):
    # Evaluates the average return on nb_runs runs of nb_episodes episodes.
    # Numbers is a list defined as [nb_runs, nb_episodes]
        nb_runs = numbers[0]
        nb_episodes = numbers[1]
       
        train_returns = [0 for i in range(nb_episodes)]
        for run in range(nb_runs):
            print("Training run:" + str(run))
            self.algo.reset()
            for episode in range(nb_episodes):
                train_returns[episode] += (self.trainOneEpisode()-train_returns[episode])/(run+1)
        return train_returns


def compare(algo_list, env_list, exp_name, nb_runs, nb_episodes, do_list = False):
# Run the combinations of algorithms in algo_list and environments in env_list as specified on do_list (all of them if do_list is False).
# Makes the average returns per episode over nb_runs runs and plots them in a folder using the experiment name.
    nb_env = len(env_list)
    nb_algo = len(algo_list)

    drawer = Drawer(exp_name)

    algo_id = 0
    env_id = 0
    legend = []
    train_returns = []


    if not do_list: ## Do all of them
        do_list = [[1 for env in env_list] for algo in algo_list]

    for algo in algo_list: # For each algo
        env_id = 0
        for env in env_list:    # For each environment
            if do_list[algo_id][env_id]:
                #print("Running on environment: " + env.name)
                legend.append(algo.name + " in " + env.name) # Add "algo in env" in the legend
                trainer = Trainer(env, algo)
                train_returns.append(trainer.evalAvgReturn([nb_runs, nb_episodes]))
            env_id += 1
        algo_id += 1

    drawer.saveMultiCSV(exp_name, train_returns, legend) # Save CSV    
    drawer.saveDetails("Experiment details", env_list, algo_list, do_list)
    return drawer, (range(nb_episodes), train_returns, "Episodes", "Average return on " + str(nb_runs), exp_name, legend) # Save return plot


if __name__ == '__main__':

    # Experiment parameters
    exp_name = "Alpha1"
    nb_runs = 1000
    nb_episodes = 300

    # Reward parameters
    rew_params1 = {"rvar_mean_ter": 10000, "rvar_var_ter": 0, "rvar_mean_step": 0, "rvar_var_step": 0}
    rew_params2 = {"rvar_mean_ter": 0, "rvar_var_ter": 0, "rvar_mean_step": 0, "rvar_var_step": 0}
    rew_params3 = {"rvar_mean_ter": 0, "rvar_var_ter": 0, "rvar_mean_step": 0, "rvar_var_step": 0}

    # Environments
    env1 = SparseTabularEnvironment(6, 6, rew_params1)
    #env2 = DenseTabularEnvironment(6, 6, rew_params2)
    #env3 = DenseTabularEnvironment(6, 6, rew_params3)

    # Algorithms
    algo_params1 = {"action_space": env1.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo_params2 = {"action_space": env1.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo1 = Sarsa(algo_params2)
    algo2 = ModifiedSarsa(algo_params2)
   
    drawer,params = compare([algo1, algo2], [env1], exp_name, nb_runs, nb_episodes)
    drawer.saveMultiPlotPNG(*params)
 