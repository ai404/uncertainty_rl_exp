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
            next_state, reward, reward_noise, rvar, done = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, reward_noise, rvar, next_state, next_action, done])
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
            #print("Training run:" + str(run))
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
    #drawer.saveMultiPlotPNG(range(nb_episodes), train_returns, "Episodes", "Average return on " + str(nb_runs), exp_name, legend) # Save return plot
    
    drawer.saveDetails("Experiment details", env_list, algo_list, do_list)
    return drawer, (range(nb_episodes), train_returns, "Episodes", "Average return on " + str(nb_runs), exp_name, legend) # Save return plot
if __name__ == '__main__':

    # Experiment parameters
    exp_name = "Third_try_j"
    nb_runs = 100
    nb_episodes = 300

    # Reward parameters
    rew_params1 = {"rvar_mean_ter": 1, "rvar_var_ter": 0.6, "rvar_mean_step": 100, "rvar_var_step": 600}
    rew_params2 = {"rvar_mean_ter": None, "rvar_var_ter": None, "rvar_mean_step": None, "rvar_var_step": None}
    rew_params3 = {"rvar_mean_ter": None, "rvar_var_ter": None, "rvar_mean_step": None, "rvar_var_step": None}

    # Environment
    env1 = SparseTabularEnvironment(6, 6, rew_params2)
    #env2 = DenseTabularEnvironment(6, 6, rew_params2)
    #env3 = DenseTabularEnvironment(6, 6, rew_params3)

    # Algorithm
    algo_params1 = {"action_space": env1.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo_params2 = {"action_space": env1.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo1 = Sarsa(algo_params2)
    algo2 = ModifiedSarsa(algo_params2)
   
    compare([algo1, algo2], [env1], exp_name, nb_runs, nb_episodes)
 