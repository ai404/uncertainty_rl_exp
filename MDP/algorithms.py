import random
from utils import *
import numpy as np
import time
from collections import defaultdict

class Agent_Q(object):
    def __init__(self, parameters):
        self.params = parameters
        self.action_space = self.params["action_space"]
        self.action_list = range(self.action_space.n)
        self.nb_actions = self.action_space.n
        self.name = "Base algo"
        self.q_values = {}
        self.default_q = 0
        self.ret = 0
        self.visits_number = {}


    def setQValue(self, state, action, value):
        if state in self.q_values:
            self.q_values[state][action] = value
        else:
            self.q_values[state] = [self.default_q for i in range(self.nb_actions)]
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


    def getVisitNumber(self, state, action):
        # Number of time this state-action pair has been visited
        assert action <= self.nb_actions, "Action " + str(action) + " is too big. Max is: " + str(self.nb_actions)
        if state in self.visits_number:
            return self.visits_number[state][action]
        else:
            return 0

    def incrementVisitNumber(self, state, action):
        # Increment the number of time this state action pair has been visited
        if state in self.visits_number:
            self.visits_number[state][action] += 1
        else:
            self.visits_number[state] = [0 for i in range(self.nb_actions)]
            self.visits_number[state][action] = 1


    def update(self, info):
        pass

    def updateNoLearn(self, reward):
        self.ret += reward

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
        self.visits_number = {}
        self.ret = 0

    def partialReset(self):
        self.ret = 0

class RandomAgent(Agent_Q):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.name = "Random"

    def nextAct(self, state):
        action = draw_softmax(self.action_list, [0 for i in range(self.nb_actions)], self.temperature)
        return action


class Sarsa(Agent_Q):
    def __init__(self, parameters):
        super().__init__(parameters)
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
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "Modified Sarsa"
        self.C = defaultdict(lambda :defaultdict(int))

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]
        #done = info[7]
        
        self.ret += R
        R = R + R_noise # Add the noise on R

        q_sa = self.getQValue(S, A)
        q_sn_an = self.getQValue(Sn, An)
        
        if R_var is None:
            wn = 10**3
        else:
            wn = 1./R_var
        self.C[S][A] = self.C[S][A] + wn
        step = self.alpha * wn/self.C[S][A]
        new_q = q_sa + step * (R + self.gamma * q_sn_an - q_sa)
        self.setQValue(S, A, new_q)


class MonteCarlo(Agent_Q):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "MonteCarlo"
        self.memory = []

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        done = info[7]

        self.ret += R
        R_seen = R + R_noise

        if not done:
            self.memory.append([S, A, R_seen])
        else:
            self.memory.append([S, A, R_seen])
            G = 0
            while len(self.memory) > 0:
                S_, A_, R_ = self.memory.pop()
                if [S_, A_, R_] in self.memory:   # First time MonteCarlo
                    G += R_
                else:
                    self.incrementVisitNumber(S_, A_)
                    C = self.getVisitNumber(S_, A_)
                    G += R_
                    q_sa = self.getQValue(S_, A_)
                    new_q = q_sa + self.alpha/C*(G - q_sa)
                    self.setQValue(S_, A_, new_q)

    def partialReset(self):
        super().partialReset()
        self.memory = []


class ModifiedMonteCarlo(Agent_Q):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "Modified MonteCarlo"
        self.memory = []
        self.rvar_memory = []
        self.C = {}

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        done = info[7]

        self.ret += R
        R_seen = R + R_noise

        if not done:
            self.memory.append([S, A, R_seen])
            self.rvar_memory.append(R_var)
        else:
            self.memory.append([S, A, R_seen])
            self.rvar_memory.append(R_var)
            G = 0
            G_var = 0
            #print(self.memory)
            #assert 0 == 1   
            while len(self.memory) > 0:
                S_, A_, R_ = self.memory.pop()          # Necessary to separate R_var from the others in order to search for first time MC
                R_var_ = self.rvar_memory.pop()
                if [S_, A_, R_] in self.memory:   # First time MonteCarlo: don't do anything if already passed here TODO: more work on this
                    G += R_
                    if R_var_:
                        G_var += R_var_
                else:
                    G += R_
                    if R_var_:
                        G_var += R_var_
                    C = self.getCValue(S_, A_)
                    if G_var:
                        w = 1./G_var
                        C += w
                    else:
                        w = 10**3
                        C += w
                    self.setCValue(S_, A_, C)
                    q_sa = self.getQValue(S_, A_)
                    new_q = q_sa + self.alpha*w/C*(G - q_sa)
                    self.setQValue(S_, A_, new_q)


    def setCValue(self, state, action, value):
        if state in self.C:
            self.C[state][action] = value
        else:
            self.C[state] = [0 for i in range(self.nb_actions)]
            self.C[state][action] = value

    def getCValue(self, state, action):
        if state in self.C:
            return self.C[state][action]
        else:
            return 0

    def partialReset(self):
        super().partialReset()
        self.memory = []

    def reset(self):
        super().reset()
        self.memory = []
        self.C = {}
























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
