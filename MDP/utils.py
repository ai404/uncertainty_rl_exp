import matplotlib.pyplot as plt
import csv
import os
import copy

import numpy as np
import math



class Drawer():
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.exp_name = exp_name
        self.makeDir("./experiments")
        self.makeDir(self.output_path_root)

    def savePickProbPNG(self, x, sum_action_step, x_label, y_label, plot_title):
        plt.subplots()
        nb_arms = len(sum_action_step)
        for i in range(nb_arms):
            y = sum_action_step[i]
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

    def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
        plt.subplots()
        for y_id in range(len(y_list)):
            y = y_list[y_id]
            if legend:
                plt.plot(x,y, label = legend[y_id])
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveCSV(self, csv_title, my_list):
        path = self.output_path_root + "/" + csv_title + ".csv"
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(my_list)
    
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


    def saveDetails(self, doc_title, list_of_env_params, list_of_algo_params, do_list = False):
        path = self.output_path_root + "/" + doc_title + ".txt"
        with open(path, "w") as file:
            file.write(self.exp_name + "\n\n")
            file.write("Environments: \n\n")
            for env_param in list_of_env_params:
                file.write(str(env_param))
            file.write("\nAlgorithms: ")
            for algo_param in list_of_algo_params:
                file.write(str(algo_param))
            if do_list:
                file.write("\nDo list: ")
                file.write(str(do_list))
                file.write("([1, 0][1, 1] means algo1: env1     algo2: env1 and env2")



def draw_softmax(list_action, list_q, temperature):
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    
    chosen_index = np.random.choice(len(list_action), 1, p=list_proba)
    return list_action[chosen_index[0]]

def draw_max(list_action, list_q):
    assert len(list_action) == len(list_q)
    chosen_index = np.argmax(list_q)
    return list_action[chosen_index]


def getSoftmaxProbas(list_action, list_q, temperature):
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    return list_proba


def drawChiSquare(mean, variance):
    basic_var = 6
    basic_mean = 3


    a = np.sqrt(variance/basic_var)
    b = mean - basic_mean*a

    assert b >= 0, "Variance and mean in drawChiSquare will lead to potential negative sampling: a = " + str(a) + " and b = " + str(b)

    return np.random.chisquare(3)*a + b      # Has a mean of 0.6, and a variance of 1/6
