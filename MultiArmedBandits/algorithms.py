import math
import numpy as np
import random

#####################################################
#                                                   #
#          Regret Minimization Algorigthms          #
#                                                   #
#####################################################

class EpsilonGreedyAlgo():

    def __init__(self, epsilon, decay_rate, omega):
        self.r = 1
        self.omega = omega
        self.nb_arms = len(self.omega)
        self.est_means = [0 for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.nb_sampled_arm = [0 for k in range(self.nb_arms)]
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.epsilon_save = epsilon
        self.minimal_epsilon = 0.01
        self.name = "egreedy"


    def reset(self):
        self.epsilon = self.epsilon_save
        self.est_means = [0 for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.nb_sampled_arm = [0 for k in range(self.nb_arms)]


    def update(self, arm_id, reward, var):
        self.nb_sampled_arm[arm_id - 1] += 1
        self.received_rewards[arm_id - 1].append(reward)
        self.est_means[arm_id - 1] += (reward - self.est_means[arm_id - 1]) / self.nb_sampled_arm[arm_id - 1]

    def nextAction(self):
        pick = np.random.rand()
        if pick < self.epsilon:
            action = random.sample(self.omega, 1)
            action = action[0]
        else:
            best_mean = max(self.est_means)
            action = self.est_means.index(best_mean) + 1

        self.epsilon = max(self.epsilon*self.decay_rate, self.minimal_epsilon)
        return action, True

    def getEstMeans(self):
        return self.est_means


class ModifiedEpsilonGreedyAlgo(EpsilonGreedyAlgo):

    def __init__(self, epsilon, decay_rate, omega):
        super().__init__(epsilon, decay_rate, omega)
        self.received_rewards_weights = [[] for k in range(self.nb_arms)]
        self.total_weight = [0 for k in range(self.nb_arms)]
        self.name = "Modified egreedy"


    def reset(self):
        EpsilonGreedyAlgo.reset(self)
        self.received_rewards_weights = [[] for k in range(self.nb_arms)]
        self.total_weight = [0 for k in range(self.nb_arms)]


    def update(self, arm_id, reward, var):
        self.received_rewards[arm_id - 1].append(reward)
        weight = 1/var
        self.received_rewards_weights[arm_id - 1].append(weight)
        self.est_means[arm_id - 1] = (self.est_means[arm_id - 1] * self.total_weight[arm_id - 1] + reward * weight)/(self.total_weight[arm_id - 1] + weight) #np.average(self.received_rewards[arm_id - 1], weights = self.received_rewards_weights[arm_id - 1])
        self.total_weight[arm_id - 1] += weight


    
class CheatingEpsilonGreedyAlgo(EpsilonGreedyAlgo):
    def __init__(self, epsilon, decay_rate, omega):
        super().__init__(epsilon, decay_rate, omega)
        self.name = "Cheating egreedy"
        print("Cheating e-greedy assumes that the best arm is the last one of the omega set. Here, it will use: " + str(max(self.omega)))

    def update(self, arm_id, reward, var):
        pass

    def nextAction(self):
        pick = np.random.rand()
        if pick < self.epsilon:
            action = random.sample(self.omega, 1)
            action = action[0]
        else:
            action = max(self.omega)

        self.epsilon = max(self.epsilon*self.decay_rate, self.minimal_epsilon)
        return action, True