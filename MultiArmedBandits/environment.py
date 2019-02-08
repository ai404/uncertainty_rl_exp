import numpy as np
import math


class BaseEnvironment():
    def __init__(self):
        self.means = []
        self.omega = {}
        self.name = ""

    def draw(self, arm_id):
        pass

    def getOmega(self):
        return self.omega.copy()    

    def getMeans(self):
        return self.means.copy()

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

    def bestArm(self):
        best_mean = max(self.means)
        best_arm = self.means.index(best_mean) + 1
        return best_arm



class ArticleEnvironment(BaseEnvironment):
    def __init__(self):
        self.means = [1, 0.8, 0.6, 0.4, 0.2, 0]
        self.variance = 0.25
        self.std_dev = math.sqrt(self.variance)
        self.omega = {1, 2, 3, 4, 5, 6}
        self.name = "Article 6-arm bandit"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance



class BookEnvironment(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.draw_variance = 1
        self.means_variance = 1
        self.name = "Sutton's book 10-arm bandit"

        self.draw_std_dev = math.sqrt(self.draw_variance)
        self.means_std_dev = math.sqrt(self.means_variance)

        self.means = []
        for i in range((len(self.omega))):
            self.means.append(np.random.normal(loc= 0, scale = self.means_std_dev))

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.draw_std_dev)
        return reward, self.draw_variance




class CertainRewardEnvironment(BaseEnvironment):
    def __init__(self):
        self.means = [0, 0.3, 0.6, 0.9]
        self.variance = 0.6
        self.std_dev = math.sqrt(self.variance)
        self.omega = {1, 2, 3, 4}
        self.name = "fixed variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance



class UncertainRewardEnvironment(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
        self.name = "changing variance 4-arm bandit - " + str(self.means)

    
    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6
        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance



class CertainRewardEnvironment2(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.variance = 0.6
        self.std_dev = math.sqrt(self.variance)
        self.means = [0, 0.05, 0.85, 0.9]
        self.name = "fixed variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1] # mu_1: 1.0    mu_2: 0.8  ... mu_6: 0.0
        reward = np.random.normal(loc=mean, scale = self.std_dev)
        return reward, self.variance




class UncertainRewardEnvironment2(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.05, 0.85, 0.9]
        self.name = "changing variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance



class UncertainRewardEnvironment3(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.45, 0.45, 0.9]
        self.name = "changing variance 4-arm bandit - "  + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance


class UncertainRewardEnvironment4(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [-1, 0.45, 0.45, 1.9]
        self.name = "changing variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance


class UncertainRewardEnvironment5(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0.4, 0.45, 0.45, 0.5]
        self.name = "changing variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance



class UncertainRewardEnvironment6(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [-0.9, 0, 0.9, 1.8]
        self.name = "changing variance 4-arm bandit - " + str(self.means)

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance



class UncertainRewardEnvironment7(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
        self.name = "changing variance 4-arm bandit - var(variance) = 0.66"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/3      # Has a mean of 1, and a variance of 2/3

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance



class UncertainRewardEnvironment8(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
        self.name = "changing variance 4-arm bandit - var(variance) = 0.24"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/5 + 0.4      # Has a mean of 1, and a variance of 0.24
        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance




class UncertainRewardEnvironment9(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
        self.name = "changing variance 4-arm bandit - var(variance) = 0.06"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/10 + 0.7      # Has a mean of 1, and a variance of 0.24

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance


class OneArmUncertainRewardEnvironment(BaseEnvironment):
    def __init__(self):
        self.means = [1]
        self.omega = {1}
        self.name = "changing variance 1-arm bandit"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/6 + 0.1      # Has a mean of 0.6, and a variance of 1/6
        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance

class TryEnvironment(BaseEnvironment):
    def __init__(self):
        self.omega = {1, 2, 3, 4}
        self.means = [0, 0.3, 0.6, 0.9]
        self.name = "Hello"

    def draw(self, arm_id):
        mean = self.means[arm_id - 1]
        variance = np.random.chisquare(3)/10 + 0.7      # Has a mean of 1, and a variance of 0.24

        reward = np.random.normal(loc=mean, scale = math.sqrt(variance))
        return reward, variance