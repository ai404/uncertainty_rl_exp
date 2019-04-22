import random
from utils import *
import numpy as np


class Agent_Q(object):
    def __init__(self, environment, parameters):
        self.env = environment
        self.params = parameters
        self.action_space = self.params["action_space"]
        self.action_list = range(self.action_space.n)
        self.nb_actions = self.action_space.n
        self.name = "Base algo"
        self.q_values = {}
        self.default_q = -100
        self.ret = 0

    def setQValue(self, state, action, value):
        if state in self.q_values:
            self.q_values[state][action] = value
        else:
            self.q_values[state] = [0 for i in range(self.nb_actions)]
            self.q_values[state][action] = value

    def getQValue(self, state, action):
        assert action <= self.nb_actions, "Action " + str(action) + " is too big. Max is: " + str(self.nb_actions)
        if state in self.q_values:
            return self.q_values[state][action]
        else:
            return self.default_q

    def getStateQValues(self, state):
        if state in self.q_values:
            return self.q_values[state]
        else:
            return [self.default_q for i in range(self.nb_actions)]

    def update(self, info):
        pass

    def updateNoLearn(self, reward):
        self.ret += reward

    def nextAct(self, state):
        return 0

    def getName(self):
        return self.name

    def getReturn(self):
        return self.ret

    def nextAct(self, state):
        state_q_vals = self.getStateQValues(state)
        action = draw_softmax(self.action_list, state_q_vals, self.temperature)
        return action

    def nextGreedyAct(self, state):
        state_q_vals = self.getStateQValues(state)
        action = draw_max(self.action_list, state_q_vals)
        return action

    def reset(self):
        self.q_values = {}
        self.ret = 0

    def partialReset(self):
        self.ret = 0

class RandomAgent(Agent_Q):
    def __init__(self, environment, parameters):
        super().__init__(environment, parameters)
        self.temperature = self.params["temperature"]
        self.name = "Random"

    def nextAct(self, state):
        action = draw_softmax(self.action_list, [0 for i in range(self.nb_actions)], self.temperature)
        return action


class Sarsa(Agent_Q):
    def __init__(self, environment, parameters):
        super().__init__(environment, parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "Sarsa"

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]

        self.ret += R
        R = R + R_noise # Add the noise on R

        q_sa = self.getQValue(S, A)
        q_sn_an = self.getQValue(Sn, An)
        new_q = q_sa + self.alpha * (R + self.gamma * q_sn_an - q_sa)
        self.setQValue(S, A, new_q)


class ModifiedSarsa(Agent_Q):
    # TODO: modify it!
    def __init__(self, environment, parameters):
        super().__init__(environment, parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "Sarsa"

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]

        self.ret += R
        R = R + R_noise # Add the noise on R

        q_sa = self.getQValue(S, A)
        q_sn_an = self.getQValue(Sn, An)
        new_q = q_sa + self.alpha * (R + self.gamma * q_sn_an - q_sa)
        self.setQValue(S, A, new_q)




















class MonteCarlo()        










































######### Other algorithms ###########

class QLearning(Agent_Q):
    def __init__(self, environment, parameters):
        super().__init__(environment, parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "QLearning"

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]

        self.ret += R
        R = R + R_noise # Add the noise on R

        q_sa = self.getQValue(S, A)
        max_q_sn_an = max(self.getStateQValues(Sn))
        new_q = q_sa + self.alpha * (R + self.gamma * max_q_sn_an - q_sa)
        self.setQValue(S, A, new_q)

class ExpectedSarsa(Agent_Q):
    def __init__(self, environment, parameters):
        super().__init__(environment, parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "ExpectedSarsa"

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]

        self.ret += R
        R = R + R_noise # Add the noise on R

        q_sa = self.getQValue(S, A)
        sn_q_vals = self.getStateQValues(Sn)
        list_proba = getSoftmaxProbas(self.action_list, sn_q_vals, self.temperature)
        exp_q_sn_an = np.average(sn_q_vals, weights = list_proba)
        new_q = q_sa + self.alpha * (R + self.gamma * exp_q_sn_an - q_sa)
        self.setQValue(S, A, new_q)
