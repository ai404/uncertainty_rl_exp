from environment import *
from algorithms import *
from utils import *
import time


class Trainer(object):
    def __init__(self, env, algo):
        self.env = env
        self.algo = algo

    def trainOneEpisode(self, render = False):
        #print(" -------- New episode -------")
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, reward_noise, reward_var, done = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, reward_noise, reward_var, next_state, next_action, done])
            state = next_state
            action = next_action
            if render: 
                env.render()
        return self.algo.getReturn()

    def evalAvgReturn(self, numbers):
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
        print("Running algo: " + algo.name)
        env_id = 0
        for env in env_list:    # For each environment
            if do_list[algo_id][env_id]:
                print("Running on environment: " + env.name)
                legend.append(algo.name + " in " + env.name) # Add "algo in env" in the legend
                trainer = Trainer(env, algo)
                train_returns.append(trainer.evalAvgReturn([nb_runs, nb_episodes]))
            env_id += 1
        algo_id += 1

    drawer.saveMultiCSV(exp_name, train_returns, legend) # Save CSV
    drawer.saveMultiPlotPNG(range(nb_episodes), train_returns, "Episodes", "Average return on " + str(nb_runs), exp_name, legend) # Save return plot
    
    algo_param_list = []
    env_param_list = []
    for algo in algo_list:
        algo_param_list.append(algo.params)
    for env in env_list:
        env_param_list.append(env.params)
    drawer.saveDetails("Experiment details", env_param_list, algo_param_list, do_list)

if __name__ == '__main__':

    # Experiment parameters
    exp_name = "Third_try_j"
    nb_runs = 5
    nb_episodes = 300

    # Reward parameters
    rew_params1 = {"reward_var_mean_ter": None, "reward_var_var_ter": None, "reward_var_mean_step": None, "reward_var_var_step": None}
    rew_params2 = {"reward_var_mean_ter": None, "reward_var_var_ter": None, "reward_var_mean_step": 1, "reward_var_var_step": 0.6}
    

    # Environment
    env1 = SemiSparseTabularEnvironment(6, 6, rew_params1)
    env2 = SemiSparseTabularEnvironment(6, 6, rew_params2)

    # Algorithm
    algo_params1 = {"action_space": env2.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo_params2 = {"action_space": env2.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo1 = MonteCarlo(algo_params1)
    algo2 = ModifiedMonteCarlo(algo_params2)
   
    compare([algo1], [env1, env2], exp_name, nb_runs, nb_episodes)
 