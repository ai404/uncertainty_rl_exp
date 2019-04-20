from environment import *
from algorithms import *
from utils import *
import time


class Trainer(object):
    def __init__(self, env, algo, drawer):
        self.env = env
        self.algo = algo
        self.drawer = drawer

    def trainOneEpisode(self, render = False):
        #print(" -------- New episode -------")
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, reward_noise, reward_var, done = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, reward_noise, reward_var, next_state, next_action])
            state = next_state
            action = next_action
            if render: 
                env.render()
        return self.algo.getReturn()


    def testOneEpisode(self, learn = False, render = False):
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextGreedyAct(state)
        if render: 
            print(" -------- New episode -------")
            env.render()
        while not done:
            next_state, reward, reward_var, done = self.env.step(action)
            if render: 
                env.render()
            next_action = self.algo.nextGreedyAct(next_state)
            if learn: self.algo.update([state, action, reward, reward_var, next_state, next_action])
            else: self.algo.updateNoLearn(reward)
            state = next_state
            action = next_action
        ret = self.algo.getReturn()
        return self.algo.getReturn()

    def evalAvgReturn(self, numbers, learn):
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
    rew_var_mean = 100000
    rew_var_var = 1
    env = SparseTabularEnvironment(6, 6, rew_var_mean, rew_var_var)
    action_space = env.action_space
    alpha = 0.3
    temperature = 1
    gamma = 1

    exp_name = "First_try"
    learn = True

    nb_runs = 500
    nb_episodes = 100

    avgs_train = []
    avgs_test = []

    algo_params = {"action_space": action_space, "temperature": temperature, "alpha":alpha, "gamma":gamma}
    algo = Sarsa(env, algo_params)

    drawer = Drawer(exp_name)
   
    trainer = Trainer(env, algo, drawer)
    train_returns = trainer.evalAvgReturn([nb_runs, nb_episodes], learn)
    
    print(train_returns)

    drawer.savePlotPNG(range(len(train_returns)), train_returns, "Episode", "Average return", "Sparse task: average return on training using algo: " + algo.getName() + ", reward var mean: " + str(rew_var_mean) + ", reward var var: " + str(rew_var_var))
