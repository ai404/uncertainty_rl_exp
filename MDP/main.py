from environments import *
from algorithms import *
from utils import *



class Trainer(object):
    def __init__(self, env, algo, drawer):
        self.env = env
        self.algo = algo
        self.drawer = drawer


    def trainOneEpisode(self):
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, done, _ = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, next_state, next_action])
            state = next_state
            action = next_action
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
            next_state, reward, done, _ = self.env.step(action)
            if render: 
                env.render()
            next_action = self.algo.nextGreedyAct(next_state)
            if learn: self.algo.update([state, action, reward, next_state, next_action])
            else: self.algo.updateNoLearn(reward)
            state = next_state
            action = next_action
        ret = self.algo.getReturn()
        return self.algo.getReturn()

 #    def evalAvgReturn(self, numbers, learn):
 #        nb_runs = numbers[0]
 #        nb_segments = numbers[1]
 #        nb_episodes = numbers[2]
 #        avg_train_ret = 0
 #        avg_test_ret = 0

 #        test_returns = []
 #        for run in range(nb_runs):
 #            print("Training run:" + str(run))

 #            self.algo.reset()
 #            for seg in range(nb_segments - 1):
 #                # Training episodes
 #                for episode in range(nb_episodes):
 #                    self.trainOneEpisode()
 #                test_returns.append(self.testOneEpisode())

 #            # Getting "training data" in the last segment
 #            for episode in range(nb_episodes):
 #                train_ret = self.trainOneEpisode()
 #                avg_train_ret += (train_ret - avg_train_ret)/(run*nb_episodes + episode + 1)


 #            # Geting "testing data" with greedy policy after training
 #            for episode in range(1):
 #                test_ret = self.testOneEpisode(learn = learn)
 #                avg_test_ret += (test_ret - avg_test_ret)/(run*10 + episode + 1)   
 #                test_returns.append(test_ret)     

	# return avg_train_ret, avg_test_ret, test_returns



if __name__ == '__main__':
	env = SparseGridWorld(4)
	action_list = env.action_list
	temperature = 1
	alpha = 0.3
	gamma = 1

	algo_params = {"action_list": action_list, "temperature": temperature, "alpha":alpha, "gamma":gamma}

	algo = RandomAgent(env, algo_params)
	env.render()


	while not env.isDone():
		env.act(algo.nextAct(env.getState))
		env.render()


