import math
import numpy as np
import random


####### Best Arm ID Algorithms ########

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


    def update(self, arm_id, reward, var):
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


    def update(self, arm_id, reward, var):
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

    def printTestBounds(self):
        curr_best_mean = max(self.est_means)
        best_mean_act = self.est_means.index(curr_best_mean) + 1
        curr_best_mean_minus_bound = curr_best_mean - self.bound(best_mean_act)

        second_best_mean_plus_bound = maxExcept(self.means_with_bounds, best_mean_act - 1)

        print("Est_means: " + str(self.est_means))
        print("Best mean act: " + str(best_mean_act))
        print("means_with_bounds: " + str(self.means_with_bounds))
        print("second_best_mean_plus_bound: " + str(second_best_mean_plus_bound))
        print("curr_best_mean_minus_bound: " + str(curr_best_mean_minus_bound))


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


    def update(self, arm_id, reward, var):
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





class ModifiedActionEliminationAlgo():
    def __init__(self, delta, epsilon, omega):
        self.r = 1
        self.omega_save = omega.copy()
        self.omega = omega
        self.nb_arms = len(self.omega)
        self.est_means = [None for k in range(self.nb_arms)]
        self.received_rewards = [[] for k in range(self.nb_arms)]
        self.received_rewards_weights = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.delta = delta
        self.epsilon = epsilon
        self.epoch_done = False
        self.epoch_k = 0
        self.epoch_list = list(self.omega)
        random.shuffle(self.epoch_list)


    def update(self, arm_id, reward, var):
        self.received_rewards[arm_id - 1].append(reward)
        weight = 0.1 + 0.15/var
        self.received_rewards_weights[arm_id - 1].append(weight)
        self.number_times_picked[arm_id - 1] += weight
        #print("Received rewards: " + str(self.received_rewards))
        #print("Received_rewards_weights: " + str(self.received_rewards_weights))
        try:
            self.est_means[arm_id - 1] = np.average(self.received_rewards[arm_id - 1], weights = self.received_rewards_weights[arm_id - 1])
        except:
            print("Received rewards: " + str(self.received_rewards))
            print("Received_rewards_weights: " + str(self.received_rewards_weights))
        
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
        self.received_rewards_weights = [[] for k in range(self.nb_arms)]
        self.number_times_picked = [0 for k in range(self.nb_arms)]
        self.epoch_done = False
        self.epoch_k = 0
        self.epoch_list = list(self.omega)
        random.shuffle(self.epoch_list)






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
        weight = 0.1 + 0.54/var
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