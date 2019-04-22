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
    # Reward parameters
    rew_var_mean_ter = 100000
    rew_var_var_ter = 1
    rew_var_mean_step = 100000
    rew_var_var_step = 1
    rew_params = {"reward_var_mean_ter": rew_var_mean_ter, "reward_var_var_ter": rew_var_var_ter, "reward_var_mean_step": rew_var_mean_step, "reward_var_var_step": rew_var_var_step}
    
    # Environment
    env = SemiSparseTabularEnvironment(6, 6, rew_params)
    action_space = env.action_space

    # Algorithm
    alpha = 0.3
    temperature = 1
    gamma = 1
    algo_params = {"action_space": action_space, "temperature": temperature, "alpha":alpha, "gamma":gamma}
    algo = Sarsa(env, algo_params)

    # Utilities
   
    # Experiment parameters
    exp_name = "Second_try"
    learn = True

    nb_runs = 500
    nb_episodes = 100

    # Experiment
    trainer = Trainer(env, algo)
    train_returns = trainer.evalAvgReturn([nb_runs, nb_episodes])
    
    # Plotting
    drawer = Drawer(exp_name)
    drawer.savePlotPNG(range(len(train_returns)), train_returns, "Episode", "Average return", env.getName() + ": return averaged on " + str(nb_runs) + " runs using " + algo.getName() + ", rew_var_mean_ter: " + str(rew_var_mean_ter) + ", rew_var_var_ter: " + str(rew_var_var_ter))
