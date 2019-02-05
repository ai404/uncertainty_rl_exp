from environment import *
from algorithms import *
from utils import smoothen
import matplotlib.pyplot as plt
import csv
import os




class MainLoop():
    def __init__(self, nb_steps, environment, algo):
        self.env = environment
        self.omega = self.env.getOmega()
        self.algo = algo
        self.algo.reset()
        self.action_memory = []
        self.reward_memory = []
        self.step_regret_memory = []
        self.total_regret_memory = []
        self.step = 0
        self.nb_steps = nb_steps

    def doOneStep(self):
        
        action_record = self.algo.nextAction()
        reward, var = self.env.draw(action_record[0])
        self.algo.update(action_record[0],reward, var)
        

        if action_record[1]:
            self.action_memory.append(action_record[0])
            self.reward_memory.append(reward)
            step_regret = self.env.bestArmMean() - reward
            self.step_regret_memory.append(step_regret)
            if len(self.total_regret_memory) > 0:
                self.total_regret_memory.append(self.total_regret_memory[-1] + step_regret)
            else:
                self.total_regret_memory.append(step_regret)
            self.step += 1


    def findBestArm(self):
        # HERE: Choose the stop condition: algorithm has solved problem OR fixed number of steps (better to do averages per step)
        if not self.nb_steps:
            while not self.algo.isDone():
                if self.step >= 10000 and self.step%500 == 0:
                        print("  Step: " + str(self.step))
                        print("Omega: " + str(self.algo.omega))
                        print("Algo_est_means: " + str(self.algo.est_means))
                        self.algo.printTestBounds()
                self.doOneStep()
        else:
            while self.step < self.nb_steps:
                self.doOneStep()

    def getActionMemory(self):
        return self.action_memory

    def getStepsUsed(self):
        return self.step

    def getTotalRegMemory(self):
        return self.total_regret_memory


class Drawer():
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.makeDir("./experiments")
        self.makeDir(self.output_path_root)

    def savePickProbPNG(self, x, sum_action_step, x_label, y_label, plot_title):
        plt.subplots()
        nb_arms = len(sum_action_step)
        for i in range(nb_arms):
            y = sum_action_step[i]
            y = smoothen(y)
            plt.plot(x, y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def savePlotPNG(self, x, y, x_label, y_label, plot_title):
        plt.subplots()
        plt.plot(x,y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveMutliPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
        plt.subplots()
        for y_id in range(len(y_list)):
            y = y_list[y_id]
            if legend:
                plt.plot(x,y, label = legend[y_id])
        if legend:
            plt.legend(legend)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveCSV(self, sum_action_step, csv_title):
        #path = self.output_path_root + "/" + csv_title + ".csv"
        #if not os.path.exists(path):
        #    with open(path, "w"):
        #        pass
        #scores_file = open(path, "a")
        #for action in sum_action_step:
        #    with scores_file: 
        #        writer = csv.writer(scores_file)
        #        writer.writerow(action)
        pass

    def makeDir(self, path):
        # Check if directory exists, if not, create it
        if not os.path.exists(path):
            try:  
                os.mkdir(path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)       

def comparePickingProbabilities(fixed_nb_h1, nb_runs, exp_name, environment, algo):
    nb_steps = int(fixed_nb_h1 * environment.getH1())
    nb_arms = environment.getNbArms()
    print("Number of steps: " + str(nb_steps))

    # Initializing the sum_action_step matrix
        # 1st index: action (1 to 6)
        # 2nd index: time step (1 to nb_steps)
        # Value inside: number of time this action has been chosen, divided by number of runs
    sum_action_step = []
    for i in range(nb_arms):
        a = ([0.0 for k in range(nb_steps)])
        sum_action_step.append(a)

    # Running the MainLoop nb_steps time and populating the sum_action_step matrix
    for i in range(nb_runs):
        main_loop = MainLoop(nb_steps, environment, algo)
        main_loop.findBestArm()
        action_mem = main_loop.getActionMemory()
        for time_step in range(len(action_mem)):
            action = action_mem[time_step]
            sum_action_step[action-1][time_step] += 1.0/nb_runs
        # Show at which step we are
        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)
    #print(str(sum_action_step)) 
    # Drawing the results
    drawer = Drawer(exp_name)
    drawer.savePickProbPNG([step/environment.getH1() for step in range(nb_steps)], sum_action_step, "Number of pulls (normalized by H1)", "P(I_t = i)", exp_name)
    drawer.saveCSV(sum_action_step, exp_name)





def averageFinishingTime(nb_runs, environment, algo):
    sum_nb_steps = 0
    n = 0

    for i in range(nb_runs):
        main_loop = MainLoop(False, environment, algo)
        main_loop.findBestArm()
        sum_nb_steps += main_loop.getStepsUsed()
        n += 1
        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)

    mean = float(sum_nb_steps)/n
    print("Average nb of step until done: " + str(mean))
    return mean



def averageRegret(nb_runs, exp_name, environment, algo, nb_steps):
    total_regrets_mean = [0 for k in range(nb_steps)]

    for i in range(nb_runs):
        main_loop = MainLoop(nb_steps, environment, algo)
        main_loop.findBestArm()
        total_regrets = main_loop.getTotalRegMemory()
        for j in range(nb_steps):
            total_regrets_mean[j] += (total_regrets[j] - total_regrets_mean[j])/(i+1)
        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)



    drawer = Drawer(exp_name)
    drawer.savePlotPNG(range(nb_steps), total_regrets_mean, "Steps", "Regret", "Average regret over " + str(nb_runs) + " runs for " + algo.name)
    print("Average total regret after " + str(nb_steps) + " steps is : " + str(total_regrets_mean[-1]))


def compareAverageRegrets(nb_runs, exp_name, environment, algo1, algo2, nb_steps):
    total_regrets_mean_1 = [0 for k in range(nb_steps)]
    total_regrets_mean_2 = [0 for k in range(nb_steps)]

    for i in range(nb_runs):
        main_loop_1 = MainLoop(nb_steps, environment, algo1)
        main_loop_2 = MainLoop(nb_steps, environment, algo2)
        main_loop_1.findBestArm()
        main_loop_2.findBestArm()
        total_regrets_1 = main_loop_1.getTotalRegMemory()
        total_regrets_2 = main_loop_2.getTotalRegMemory()
        for j in range(nb_steps):
            total_regrets_mean_1[j] += (total_regrets_1[j] - total_regrets_mean_1[j])/(i+1)
            total_regrets_mean_2[j] += (total_regrets_2[j] - total_regrets_mean_2[j])/(i+1)

        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)

    both_reg_means = [total_regrets_mean_1, total_regrets_mean_2]

    drawer = Drawer(exp_name)
    drawer.saveMutliPlotPNG(range(nb_steps), both_reg_means, "Steps", "Regret", "Comparison of average regrets over " + str(nb_runs) + " runs", [algo1.name, algo2.name])
    print(str(algo1.name) + ": average total of regret after " + str(nb_steps) + " steps and " + str(nb_runs) + " runs is : " + str(total_regrets_mean_1[-1]))
    print(str(algo2.name) + ": average total of regret after " + str(nb_steps) + " steps and " + str(nb_runs) + " runs is : " + str(total_regrets_mean_2[-1]))



if __name__ == "__main__":

    NB_RUNS = 2000
    EXP_NAME = "Try_e_05_dec_0995_corrected"
    ENVIRONMENT = UncertainRewardEnvironment()
    TASK = "RegretMin"  # "BestArmIDPickProb" "BestArmIDFinTime"


    # Best arm identification parameters
    if TASK == "BestArmIDPickProb":
        DELTA = 0.1
        EPSILON = 0.01
        FIXED_NB_H1 = 80
        ALGO = ModifiedActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())
        comparePickingProbabilities(FIXED_NB_H1, NB_RUNS, EXP_NAME, ENVIRONMENT, ALGO)

    elif TASK == "BestArmIDFinTime":
        DELTA = 0.1
        EPSILON = 0.01
        ALGO = ModifiedActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())
        averageFinishingTime(NB_RUNS, ENVIRONMENT, ALGO)

    elif TASK == "RegretMin":
        EPSILON = 0.5
        DECAY = 0.995
        NB_STEPS = 1000
        ALGO1 = EpsilonGreedyAlgo(EPSILON, DECAY, ENVIRONMENT.getOmega())
        ALGO2 = ModifiedEpsilonGreedyAlgo(EPSILON, DECAY, ENVIRONMENT.getOmega())
        compareAverageRegrets(NB_RUNS, EXP_NAME, ENVIRONMENT, ALGO1, ALGO2, NB_STEPS)
    # Regret 



