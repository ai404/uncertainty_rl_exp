import random
from utils import *
import numpy as np
import time
from collections import defaultdict
from environment import bdone

class Agent_Q(object):
# Parent class for Q-learning agents
    def __init__(self, parameters):
        self.params = parameters                            # Parameters used by the algorithm
        self.action_space = self.params["action_space"]     # Action space (gym.spaces.Discrete())
        self.action_list = range(self.action_space.n)       # List of possible actions
        self.nb_actions = self.action_space.n               # Number of actions
        self.name = "Base algo"                             # Algo name
        self.q_values = {}                                  # Q values dictionnary (for each S-A pair)
        self.default_q = 0                                  # Default Q value
        self.ret = 0                                        # Noiseless return
        self.visits_number = {}                             # Number of visits (for each S-A pair)


    def setQValue(self, state, action, value):
    # Set the Q value for a state action pair
        if state in self.q_values:
            self.q_values[state][action] = value
        else:
            self.q_values[state] = [self.default_q for i in range(self.nb_actions)]
            self.q_values[state][action] = value

    def getQValue(self, state, action):
    # Get the Q value for a state action pair
        assert action <= self.nb_actions, "Action " + str(action) + " is too big. Max is: " + str(self.nb_actions)
        if state in self.q_values:
            return self.q_values[state][action]
        else:
            return self.default_q

    def getStateQValues(self, state):
    # Get a list of all the action values for a given state
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
    # Update the q-values given the info list.
    # info has the following format: [State_t, Action_t, Reward_t, Reward_noise_t, Reward_variance_t, State_t+1, Action_t+1, EpisodeDone]
        pass

    def updateNoLearn(self, reward):
    # Update without learning (just adds the reward to the full return)
        self.ret += reward

    def getName(self):
        return self.name

    def getReturn(self):
        return self.ret

    def nextAct(self, state):
    # Chooses the next action given a state using the softmax 
        state_q_vals = self.getStateQValues(state)
        action = draw_softmax(self.action_list, state_q_vals, self.temperature)
        return action

    def nextGreedyAct(self, state):
    # Chooses the next action given a state using a greedy policy (action with best q value)
        state_q_vals = self.getStateQValues(state)
        action = draw_max(self.action_list, state_q_vals)
        return action

    def reset(self):
    # Resets the agent completely (for a new run)
        self.q_values = {}
        self.visits_number = {}
        self.ret = 0

    def partialReset(self):
    # Partially resets the agent for a new episode
        self.ret = 0

class RandomAgent(Agent_Q):
# Completely random agent (no update)
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.name = "Random"

    def nextAct(self, state):
        action = draw_softmax(self.action_list, [0 for i in range(self.nb_actions)], self.temperature)
        return action


class Sarsa(Agent_Q):
# Classical Sarsa agent
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

        self.ret += R                                                       # Add noiseless R to the noiseless return
        R = R + R_noise                                                     # Add the noise on R
        q_sa = self.getQValue(S, A)                                         # Get Q(S_t,A_t)
        q_sn_an = self.getQValue(Sn, An)                                    # Get Q(S_t+1, A_t+1)
        self.incrementVisitNumber(S, A)                                     # Increment the number of visits to S, A
        C = self.getVisitNumber(S, A)                                       # Get the number of visits to S, A
        new_q = q_sa + self.alpha/C* (R + self.gamma * q_sn_an - q_sa)      # Compute new Q(S_t, A_t) based on unweighted average update rule
        self.setQValue(S, A, new_q)                                         # Save new Q value for S_t, A_t


class ModifiedSarsa(Agent_Q):
# Modified Sarsa agent with inverse variance weighting
    def __init__(self, parameters):
        super().__init__(parameters)
        self.temperature = self.params["temperature"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]
        self.default_q = 0
        self.name = "Modified Sarsa"
        self.C = {}                                                         # C values (C is the sum of weights w)

    def update(self, info):
        S = info[0]
        A = info[1]
        R = info[2]
        R_noise = info[3]
        R_var = info[4]
        Sn = info[5]
        An = info[6]
        done = info[7]
        
        self.ret += R                                                       # Add noiseless R to the noiseless return
        R = R + R_noise                                                     # Add the noise on R

        q_sa = self.getQValue(S, A)                                         # Get Q(S_t, A_t)
        q_sn_an = self.getQValue(Sn, An)                                    # Get Q(S_t+1, A_t+1)
        
        if R_var is None:                                                   # Minimal variance on the reward
            R_var = 1./10**3
        if done == bdone.TERM:                                              # We are sure of the terminal state being the terminal state with 0 Q-values!
            for an in self.action_list:                                     # For all actions of the terminal state
                self.setCValue(Sn, an, 10**3)                               # Set C = 1000 (which is like saying that we have a 1/1000 variance on the Q-values for terminal state)
        C_SnAn = self.getCValue(Sn, An)                                     # Get C(S_t+1, A_t+1)
        wn = 1./(R_var + (self.gamma**2)/C_SnAn)                            # Compute weight according to update rule

        C_SA = self.getCValue(S, A)                                         # Get C(S,A)
        C_SA += wn                                                          # Increment C(S,A) with the weight
        self.setCValue(S, A, C_SA)                                          # Save the new value of C(S,A)
        step = self.alpha * wn/C_SA                                         # Compute step value
        new_q = q_sa + step * (R + self.gamma * q_sn_an - q_sa)             # Update Q(S,A) with inverse-variance weighted average
        self.setQValue(S, A, new_q)                                         # Save new Q(S,A)

    def reset(self):
        super().reset()
        self.C = {}                                                         # Delete the C values when completely 

    def setCValue(self, state, action, value):
    # Set C value for given state-action pair
        if state in self.C:
            self.C[state][action] = value
        else:
            self.C[state] = [1./10**6 for i in range(self.nb_actions)]
            self.C[state][action] = value

    def getCValue(self, state, action):
    # Get C value of given state-action pair
        if state in self.C:
            return self.C[state][action]
        else:
            return 1./10**6


class MonteCarlo(Agent_Q):
# Classical Monte Carlo agent
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

        self.ret += R                                                           # Add noiseless R to the noiseless return 
        R_seen = R + R_noise                                                    # Add noise on reward

        if not done:                                                            # Add state, action, reward in memory
            self.memory.append([S, A, R_seen])
        else:
            self.memory.append([S, A, R_seen])
            G = 0 
            while len(self.memory) > 0:                                         # Recursively reading the memory
                S_, A_, R_ = self.memory.pop()
                if [S_, A_, R_] in self.memory:                                 # First time MonteCarlo: don't update if same S_, A_, R_ in memory (should have been only S_, A_ but that's not how the experiments were run. TODO)    
                    G += R_                                                     # (still increment the return)
                else:
                    self.incrementVisitNumber(S_, A_)                           # Increment visit number 
                    C = self.getVisitNumber(S_, A_)                             # Get visit number
                    G += R_                                                     # Increment return with reward
                    q_sa = self.getQValue(S_, A_)                               # Get Q(S,A)
                    new_q = q_sa + self.alpha/C*(G - q_sa)                      # Update Q(S,A) using weighted update rule
                    self.setQValue(S_, A_, new_q)                               # Save new Q(S,A)

    def partialReset(self):                                                     # When starting new episode
        super().partialReset()
        self.memory = []                                                        # Clear memory (if not already all popped)

    def reset(self):                                                            # When starting a new run
        super().reset()
        self.memory = []                                                        # Clear memory (if not already all popped)

class ModifiedMonteCarlo(Agent_Q):
# Modified Monte Carlo agent with inverse variance weighting 
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
                S_, A_, R_ = self.memory.pop()                                  # Necessary to separate R_var from the others in order to search for first time MC
                R_var_ = self.rvar_memory.pop()
                if [S_, A_, R_] in self.memory:                                 # First time MonteCarlo: don't update if same S_, A_, R_ in memory (should have been only S_, A_ but that's not how the experiments were run. TODO)
                    G += R_                                                     # Increment seen return with noisy reward
                    if R_var_:                                                  # If there is a variance on the reward
                        G_var += R_var_                                         # Increment return variance (sum of reward variances)
                else:           
                    G += R_                                                     # Increment seen return with noisy reward
                    if R_var_:                                                  # If there is a variance on the reward
                        G_var += R_var_                                         # Increment return variance (sum of reward variances)
                    C = self.getCValue(S_, A_)                                  # Get C(S_, A_)
                    if G_var:                                                   # If G_var is not 0
                        w = 1./G_var                                            # Compute weight as 1/G_var
                        C += w                                                  # Increment C(S_, A_)
                    else:                                                       # Else if G_var is 0
                        w = 10**3                                               # By default give a value to w as 10^3 (as if G is very certain)
                        C += w                                                  # And increment C(S_, A_)
                    self.setCValue(S_, A_, C)                                   # Save C(S_, A_)
                    q_sa = self.getQValue(S_, A_)                               # Get Q(S_, A_)
                    new_q = q_sa + self.alpha*w/C*(G - q_sa)                    # Update Q(S_, A_) according to inverse variance update rule
                    self.setQValue(S_, A_, new_q)                               # Save Q(S_, A_)


    def setCValue(self, state, action, value):
    # Set C value for given state-action pair
        if state in self.C:
            self.C[state][action] = value
        else:
            self.C[state] = [0 for i in range(self.nb_actions)]
            self.C[state][action] = value

    def getCValue(self, state, action):
    # Get C value for a given state action pair
        if state in self.C:
            return self.C[state][action]
        else:
            return 0

    def partialReset(self):                                                     # When starting a new episode
        super().partialReset()
        self.memory = []                                                        # Clear memory (if not already all popped

    def reset(self):                                                            # When starting a new run
        super().reset() 
        self.memory = []                                    
        self.C = {}                                                             # Clear C values
























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
