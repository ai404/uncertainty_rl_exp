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



if __name__ == '__main__':

    # Experiment parameters
    exp_name = "Third_try_b"
    nb_runs = 50
    nb_episodes = 300

    # Reward parameters
    rew_params1 = {"reward_var_mean_ter": 10000, "reward_var_var_ter": None, "reward_var_mean_step": None, "reward_var_var_step": None}
    rew_params2 = {"reward_var_mean_ter": None, "reward_var_var_ter": None, "reward_var_mean_step": 1, "reward_var_var_step": None}
    

    # Environment
    env1 = SemiSparseTabularEnvironment(6, 6, rew_params1)
    env2 = SemiSparseTabularEnvironment(6, 6, rew_params2)

    # Algorithm
    algo_params = {"action_space": env1.action_space, "temperature": 1, "alpha": 0.3, "gamma": 1}
    algo = MonteCarlo(algo_params)
   
    # Experiment
    trainer1 = Trainer(env1, algo)
    train_returns1 = trainer1.evalAvgReturn([nb_runs, nb_episodes])
    trainer2 = Trainer(env2, algo)
    train_returns2 = trainer2.evalAvgReturn([nb_runs, nb_episodes])  

    # Plotting
    drawer = Drawer(exp_name)

    legend =  [str(rew_params1), str(rew_params2)]
    drawer.saveMultiPlotPNG(range(len(train_returns1)), [train_returns1, train_returns2], "Episode", "Average return", env1.getName() + ": return averaged on " + str(nb_runs) + " runs using " + algo.getName(), legend)
    drawer.saveMultiCSV(env1.getName() + ": return averaged on " + str(nb_runs) + " runs using " + algo.getName(), [train_returns1, train_returns2], legend)