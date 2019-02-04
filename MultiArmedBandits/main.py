from environment import *
from algorithms import ActionEliminationAlgo, UCBAlgo, LUCBAlgo, ModifiedActionEliminationAlgo
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
        self.step = 0
        self.nb_steps = nb_steps

    def doOneStep(self):
        
        action_record = self.algo.nextAction()
        reward, var = self.env.draw(action_record[0])
        self.algo.update(action_record[0],reward, var)

        if action_record[1]:
            self.action_memory.append(action_record[0])
            self.reward_memory.append(reward)
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
            return self.algo.result()
        else:
            while self.step < self.nb_steps:
                self.doOneStep()
            return self.algo.result()

    def get_action_memory(self):
        return self.action_memory

    def getStepsUsed(self):
        return self.step


class Drawer():
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.make_dir("./experiments")
        self.make_dir(self.output_path_root)

    def save_pick_prob_png(self, x, sum_action_step, x_label, y_label, plot_title):
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

    def save_csv(self, sum_action_step, csv_title):
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

    def make_dir(self, path):
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
        result = main_loop.findBestArm()
        action_mem = main_loop.get_action_memory()
        for time_step in range(len(action_mem)):
            action = action_mem[time_step]
            sum_action_step[action-1][time_step] += 1.0/nb_runs
        # Show at which step we are
        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)
    #print(str(sum_action_step)) 
    # Drawing the results
    drawer = Drawer(exp_name)
    drawer.save_pick_prob_png([step/environment.getH1() for step in range(nb_steps)], sum_action_step, "Number of pulls (normalized by H1)", "P(I_t = i)", exp_name)
    drawer.save_csv(sum_action_step, exp_name)

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



if __name__ == "__main__":

    NB_RUNS = 500
    FIXED_NB_H1 = 80
    EXP_NAME = "Try1"

    DELTA = 0.1
    EPSILON = 0.01


    ENVIRONMENT = UncertainRewardEnvironment()
    ALGO = ModifiedActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())

    #comparePickingProbabilities(FIXED_NB_H1, NB_RUNS, EXP_NAME, ENVIRONMENT, ALGO)
    averageFinishingTime(NB_RUNS, ENVIRONMENT, ALGO)