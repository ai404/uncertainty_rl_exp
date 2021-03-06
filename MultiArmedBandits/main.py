from environment import *
from algorithms import *
from utils import *
import matplotlib.pyplot as plt
import csv
import os
import copy




class MainLoop():
    def __init__(self, nb_steps, environment, algo):
        self.env = environment
        self.omega = self.env.getOmega()
        self.algo = algo
        self.algo.reset()
        self.action_memory = []
        self.reward_memory = []
        self.est_means_err_memory = []
        self.step_regret_memory = []
        self.total_regret_memory = []
        self.step = 0
        self.nb_steps = nb_steps
        self.best_move_memory = []

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
            if action_record[0] == self.env.bestArm():
                self.best_move_memory.append(1)
            else:
                self.best_move_memory.append(0)
            real_means = self.env.getMeans()
            est_means = self.algo.getEstMeans()
            L2_mean_error = L2_distance(real_means, est_means)
            self.est_means_err_memory.append(L2_mean_error)
            self.step += 1


    def runLoop(self):
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

    def getBestMoveMemory(self):
        return self.best_move_memory

    def getEstMeanErrMemory(self):
        return self.est_means_err_memory

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

    def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False, ylimit = False):
        plt.subplots()
        for y_id in range(len(y_list)):
            y = y_list[y_id]
            y_smooth = smoothen(y)
            if legend:
                plt.plot(x,y, label = legend[y_id])
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
        if ylimit:
            plt.ylim(top=ylimit)  # adjust the top leaving bottom unchanged
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
    
    def saveMultiCSV(self, csv_title, list_of_lists, legend):
        path = self.output_path_root + "/" + csv_title + ".csv"
        assert len(legend) == len(list_of_lists), "Error writing CSV " + csv_title + ": legend has to have same size than list_of_lists"
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            i = 0
            for a_list in list_of_lists:
                b_list = [legend[i]] + a_list
                writer.writerow(b_list)
                i += 1

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
        main_loop.runLoop()
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
        main_loop.runLoop()
        sum_nb_steps += main_loop.getStepsUsed()
        n += 1
        if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0):
            print(i * 100 / nb_runs)

    mean = float(sum_nb_steps)/n
    print("Average nb of step until done: " + str(mean))
    return mean

def compareAlgosEnv(nb_runs, exp_name, env_list, algo_list, nb_steps, do_list = False):
    ###
    # Plots the average regret, best move probability, and mean estimate error

    # do_list is a 1 or 0 list saying which combination should be done. 
    # do_list[i][j] = 1 means we want algo [i] in env[j] to be tested

    nb_env = len(env_list)
    nb_algo = len(algo_list)

    drawer = Drawer(exp_name)

    nb_experiments = sum(sum(do) for do in do_list)

    all_reg_means = [[0 for step in range(nb_steps)] for exp in range(nb_experiments)]
    best_move_prob = [[0 for step in range(nb_steps)] for exp in range(nb_experiments)]
    all_means_errors = [[0 for step in range(nb_steps)] for exp in range(nb_experiments)]

    exp_id = 0
    algo_id = 0
    env_id = 0
    legend = []

    if not do_list: ## Do all of them
        do_list = [[1 for env in env_list] for algo in algo_list]

    for algo in algo_list: # For each algo
        print("Running algo: " + algo.name)
        env_id = 0
        for env in env_list:    # For each environment
            if do_list[algo_id][env_id]:
                print("Running on environment: " + env.name)
                legend.append(algo.name + " in " + env.name) # Add "algo in env" in the legend
                for i in range(nb_runs):                     # Starting the run
                    main_loop = MainLoop(nb_steps, env, algo)   # Start main loop
                    main_loop.runLoop()                     # Run until nb_steps is achieved
                    total_regrets = main_loop.getTotalRegMemory()   # Get regret memory
                    best_move = main_loop.getBestMoveMemory()
                    mean_error_mem = main_loop.getEstMeanErrMemory()   # Get regret memory
                    for j in range(nb_steps):                   # For each step
                        all_reg_means[exp_id][j] += (total_regrets[j] - all_reg_means[exp_id][j])/(i+1) # Update the regret mean
                        best_move_prob[exp_id][j] += (best_move[j] - best_move_prob[exp_id][j])/(i+1)   # Update the best move probability
                        all_means_errors[exp_id][j] += (mean_error_mem[j] - all_means_errors[exp_id][j])/(i+1) # Update the regret mean
                    if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0): # Printing stuff to see where it is
                        print(i * 100 / nb_runs)
                print(str(env.name) + ": average total of regret after " + str(nb_steps) + " steps and " + str(nb_runs) + " runs is : " + str(all_reg_means[exp_id][-1]))
                exp_id += 1
            env_id += 1
        algo_id += 1

    drawer.saveMultiCSV(exp_name + "_avg_regret", all_reg_means, legend) # Save CSV
    drawer.saveMultiCSV(exp_name + "_best_move_prob", best_move_prob, legend) # Save CSV
    drawer.saveMultiCSV(exp_name + "_mean_est_error", all_means_errors, legend) # Save CSV
    drawer.saveMultiPlotPNG(range(nb_steps), all_reg_means, "steps", "Regret", "Comparison of average regrets over " + str(nb_runs), legend) # Save regret plot
    drawer.saveMultiPlotPNG(range(nb_steps), best_move_prob, "steps", "Probability of best move", "Best move probability over " + str(nb_runs), legend) # Save best move proba plot
    drawer.saveMultiPlotPNG(range(nb_steps), all_means_errors, "steps", "L2 error on estimated means", "Average error on estimated means on " + str(nb_runs), legend, 0.5) # Save error plot


def estimateMeanError(nb_runs, exp_name, env_list, algo_list, nb_steps, do_list = False):
    nb_env = len(env_list)
    nb_algo = len(algo_list)

    drawer = Drawer(exp_name)

    nb_experiments = sum(sum(do) for do in do_list)

    all_means_errors = [[0 for step in range(nb_steps)] for exp in range(nb_experiments)]
    exp_id = 0
    algo_id = 0
    env_id = 0
    legend = []

    if not do_list: ## Do all of them
        do_list = [[1 for env in env_list] for algo in algo_list]

    for algo in algo_list: # For each algo
        print("Running algo: " + algo.name)
        env_id = 0
        for env in env_list:    # For each environment
            if do_list[algo_id][env_id]:
                print("Running on environment: " + env.name)
                legend.append(algo.name + " in " + env.name) # Add "algo in env" in the legend
                for i in range(nb_runs):                     # Starting the run
                    main_loop = MainLoop(nb_steps, env, algo)   # Start main loop
                    main_loop.runLoop()                     # Run until nb_steps is achieved
                    mean_error_mem = main_loop.getEstMeanErrMemory()   # Get regret memory
                    for j in range(nb_steps):                   # For each step
                        all_means_errors[exp_id][j] += (mean_error_mem[j] - all_means_errors[exp_id][j])/(i+1) # Update the regret mean
                    if (nb_runs >= 1000 and i % (nb_runs/500) == 0) or (nb_runs <= 1000 and i % (nb_runs/10) == 0): # Printing stuff to see where it is
                        print(i * 100 / nb_runs)
                #all_means_errors[exp_id][0] = all_means_errors[exp_id][1]
                exp_id += 1
            env_id += 1
        algo_id += 1

    drawer.saveMultiCSV(exp_name + "_mean_est_error", all_means_errors, legend) # Save CSV
    drawer.saveMultiPlotPNG(range(nb_steps), all_means_errors, "steps", "L2 error on estimated means", "Average error on estimated means on " + str(nb_runs), legend) # Save regret plot

if __name__ == "__main__":

    NB_RUNS = 5000
    EXP_NAME = "19-03-19_5_b"
    TASK = "AlgoEnvCompare"  # "BestArmIDPickProb" "BestArmIDFinTime" "AlgoEnvCompare" "MeanEstError"


    # Task definitions:
    if TASK == "BestArmIDPickProb":
        ENVIRONMENT = CertainRewardEnvironment()
        DELTA = 0.1
        EPSILON = 0.01
        FIXED_NB_H1 = 80
        ALGO = ModifiedActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())
        comparePickingProbabilities(FIXED_NB_H1, NB_RUNS, EXP_NAME, ENVIRONMENT, ALGO)

    elif TASK == "BestArmIDFinTime":
        ENVIRONMENT = CertainRewardEnvironment()
        DELTA = 0.1
        EPSILON = 0.01
        ALGO = ModifiedActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())
        averageFinishingTime(NB_RUNS, ENVIRONMENT, ALGO)

    elif TASK == "AlgoEnvCompare":
        ENV1 = UncertainRewardEnvironment7()
        ENV2 = UncertainRewardEnvironment8()
        ENV3 = UncertainRewardEnvironment9()
        EPSILON = 0.5
        DECAY = 0.995
        NB_STEPS = 1000
        ALGO0 = EpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        ALGO1 = ModifiedEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        ALGO2 = CheatingEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        #ALGO1 = ModifiedEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        #ALGO2 = CheatingEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())

        DO_LIST = [[0, 0, 1], [1, 1, 1], [1, 0, 0]]

        compareAlgosEnv(NB_RUNS, EXP_NAME, [ENV1, ENV2, ENV3], [ALGO0, ALGO1, ALGO2], NB_STEPS, DO_LIST)

    elif TASK == "MeanEstError":
        ENV1 = UncertainRewardEnvironment()
        #ENV2 = UncertainRewardEnvironment2()
        #ENV3 = UncertainRewardEnvironment3()
        #ENV4 = UncertainRewardEnvironment2()
        #ENV3 = UncertainRewardEnvironment9()
        EPSILON = 0.5
        DECAY = 0.995
        NB_STEPS = 1000
        ALGO0 = EpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        ALGO1 = ModifiedEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())
        #ALGO2 = CheatingEpsilonGreedyAlgo(EPSILON, DECAY, ENV1.getOmega())

        DO_LIST = [[1], [1]]

        estimateMeanError(NB_RUNS, EXP_NAME, [ENV1], [ALGO0, ALGO1], NB_STEPS, DO_LIST)
