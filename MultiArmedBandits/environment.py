import numpy as np
import math




class ArticleEnvironment():
    def __init__(self):
        self.means = [1, 0.8, 0.6, 0.4, 0.2, 0]
        self.variance = 0.25
        self.std_dev = math.sqrt(self.variance)
        self.omega = {1, 2, 3, 4, 5, 6}

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance


    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1


    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean





class BookEnvironment():
    def __init__(self):
        self.omega = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.draw_variance = 1
        self.means_variance = 1

        self.draw_std_dev = math.sqrt(self.draw_variance)
        self.means_std_dev = math.sqrt(self.means_variance)

        self.means = []
        for i in range((len(self.omega))):
            self.means.append(np.random.normal(loc= 0, scale = self.means_std_dev))

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.draw_std_dev)
        return reward, self.draw_variance

    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1

    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean





class CertainRewardEnvironment():
    def __init__(self):
        self.means = [0, 0.3, 0.6, 0.9]
        self.variance = 0.6
        self.std_dev = math.sqrt(self.variance)
        self.omega = {1, 2, 3, 4}

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance


    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1


    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean


class UncertainRewardEnvironment():
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
    
    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6
        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance

    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1


    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean



class CertainRewardHardEnvironment():
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.variance = 0.6
        self.std_dev = math.sqrt(self.variance)
        self.means = [0, 0.80, 0.85, 0.9]
    
    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance

    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1


    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean


class UncertainRewardHardEnvironment():
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.80, 0.85, 0.9]
    
    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance

    def getOmega(self):
        return self.omega.copy()

    def getH1(self):
        H1 = 0
        for mean in self.means:
            if mean != max(self.means):
                H1 += (max(self.means) - mean)**(-2)
        return H1


    def getNbArms(self):
        return len(self.omega)

    def bestArmMean(self):
        best_mean = max(self.means)
        return best_mean

