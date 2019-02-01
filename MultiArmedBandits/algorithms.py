from utils import CFunction, UFunction, secondLargest, maxExcept
import math
import numpy as np
import random


class ActionEliminationAlgo():
    def __init__(self, delta, epsilon, omega):
        self.r = 1
        self.omega_save = omega.copy()
        self.omega = omega
        self.nb_arms = len(self.omega)
        self.est_means = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.delta = delta
        self.epsilon = epsilon
        self.epoch_done = False
        self.epoch_k = 0
        self.epoch_list = list(self.omega)
        random.shuffle(self.epoch_list)


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = np.mean(self.received_rewards[arm_id - 1])
        
        if self.epoch_done:
            self.removeBadArms()
            self.epoch_done = False
            self.epoch_k = 0
            self.epoch_list = list(self.omega)
            random.shuffle(self.epoch_list)


    def bound(self, arm_id):
        bound = CFunction(self.number_times_picked[arm_id - 1], self.delta, self.nb_arms, self.epsilon)
        return bound


    def removeBadArms(self):
        if len(self.omega) > 1:
            est_means_with_bounds = [-math.inf for k in range(self.nb_arms)]
            for element in self.omega:
                est_means_with_bounds[element-1] = self.est_means[element-1] + self.bound(element)
            curr_best = max(est_means_with_bounds)
            curr_best_arm_id = est_means_with_bounds.index(curr_best) + 1
            curr_best_minus_bound = curr_best - 2*self.bound(curr_best_arm_id)

            for arm_id in self.epoch_list:
                if curr_best_minus_bound > est_means_with_bounds[arm_id - 1]: 
                    self.omega.remove(arm_id)
        else:
            pass


    def nextAction(self):
        if len(self.epoch_list) == 1:
            return self.epoch_list[0], True
        else:
            action = self.epoch_list[self.epoch_k % len(self.epoch_list)]
            self.epoch_k += 1
            if self.epoch_k >= len(self.omega) * self.r:
                self.epoch_done = True
                #print("Epoch done! ")
            record = True
            return action, True


    def isDone(self):
        if len(self.omega) == 1:
            return True
        else:
            return False

    def result(self):
        return list(self.omega)[0]

    def reset(self):
        self.omega = self.omega_save.copy()
        self.nb_arms = len(self.omega)
        self.est_means = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.epoch_done = False
        self.epoch_k = 0
        self.epoch_list = list(self.omega)
        random.shuffle(self.epoch_list)




class UCBAlgo():
    def __init__(self, delta, epsilon, omega):
        self.omega = omega
        self.nb_arms = len(self.omega)
        self.est_means = [None for k in range(self.nb_arms)]
        self.means_with_bounds = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.delta = delta
        self.epsilon = epsilon
        self.omega_list = list(self.omega)
        self.initialisation_done = False
        self.initial_count = 0
        self.is_done = False
        self.end_result = None
        self.beta = 1.66


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = np.mean(self.received_rewards[arm_id - 1])

    def nextAction(self):
        if self.isDone():
            return self.end_result
            record = True
        else:
            if not self.initialisation_done:
                action = self.omega_list[self.initial_count]
                self.initial_count += 1
                if self.initial_count >= len(self.omega_list):
                    self.initialisation_done = True
                record = False
            else:
                for i in range(len(self.omega)):
                    self.means_with_bounds[i] = self.est_means[i] + self.bound(i+1)
                curr_best_mean_with_bounds = max(self.means_with_bounds)
                action_id = self.means_with_bounds.index(curr_best_mean_with_bounds)
                action = self.omega_list[action_id]
                record = True
            return action, record

    def bound(self, arm_id):
        bound = (1+self.beta)*CFunction(self.number_times_picked[arm_id - 1], self.delta, self.nb_arms, self.epsilon)
        return bound


    def isDone(self):

        if None in self.est_means or None in self.means_with_bounds:
            return False

        if self.is_done:
            return True

        curr_best_mean = max(self.est_means)
        best_mean_act = self.est_means.index(curr_best_mean) + 1
        curr_best_mean_minus_bound = curr_best_mean - self.bound(best_mean_act)

        second_best_mean_plus_bound = maxExcept(self.means_with_bounds, best_mean_act - 1)

        if curr_best_mean_minus_bound > second_best_mean_plus_bound:
            self.is_done = True
            self.end_result = best_mean_act
            return True
        else:
            return False

    def result(self):
        return self.end_result

    def reset(self):
        self.est_means = [None for k in range(self.nb_arms)]
        self.means_with_bounds = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.initialisation_done = False
        self.initial_count = 0
        self.is_done = False
        self.end_result = None



class LUCBAlgo():
    def __init__(self, delta, epsilon, omega):
        self.omega = omega
        self.nb_arms = len(self.omega)
        self.est_means = [None for k in range(self.nb_arms)]
        self.means_with_bounds = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.delta = delta
        self.epsilon = epsilon
        self.omega_list = list(self.omega)
        self.initialisation_done = False
        self.initial_count = 0
        self.is_done = False
        self.end_result = None
        self.one_two_switch = 0


    def update(self, arm_id, reward):
        self.received_rewards[arm_id - 1].append(reward)
        self.number_times_picked[arm_id - 1] += 1
        self.est_means[arm_id - 1] = np.mean(self.received_rewards[arm_id - 1])

    def nextAction(self):
        if self.isDone():
            record = True
            return self.end_result, record
        else:
            if not self.initialisation_done:
                action = self.omega_list[self.initial_count]
                self.initial_count += 1
                if self.initial_count >= len(self.omega_list):
                    self.initialisation_done = True
                record = False
            else:
                if self.one_two_switch == 0:
                    curr_best_mean = max(self.est_means)
                    action_id = self.est_means.index(curr_best_mean)
                    action = self.omega_list[action_id]
                    self.one_two_switch = 1
                    record = True
                else:
                    curr_best_mean = max(self.est_means)
                    best_id = self.est_means.index(curr_best_mean)
                    for i in range(len(self.omega)): # Only update self.means_with_bounds when we receive the first one
                        self.means_with_bounds[i] = self.est_means[i] + self.bound(i+1)
                    best_mean_plus_bound =  maxExcept(self.means_with_bounds, best_id)
                    action_id = self.means_with_bounds.index(best_mean_plus_bound)
                    action = self.omega_list[action_id]
                    self.one_two_switch = 0
                    record = True
            return action, record

    def bound(self, arm_id):
        bound = CFunction(self.number_times_picked[arm_id - 1], self.delta, self.nb_arms, self.epsilon)
        return bound


    def isDone(self):
        if None in self.est_means or None in self.means_with_bounds:
            return False

        if self.is_done:
            return True

        curr_best_mean = max(self.est_means)
        best_mean_act = self.est_means.index(curr_best_mean) + 1
        curr_best_mean_minus_bound = curr_best_mean - self.bound(best_mean_act)


        second_best_mean_plus_bound = maxExcept(self.means_with_bounds, best_mean_act - 1)

        if curr_best_mean_minus_bound > second_best_mean_plus_bound:
            self.is_done = True
            self.end_result = best_mean_act
            return True
        else:
            return False

    def result(self):
        return self.end_result

    def reset(self):
        self.est_means = [None for k in range(self.nb_arms)]
        self.means_with_bounds = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.omega_list = list(self.omega)
        self.initialisation_done = False
        self.initial_count = 0
        self.is_done = False
        self.end_result = None
        self.one_two_switch = 0
