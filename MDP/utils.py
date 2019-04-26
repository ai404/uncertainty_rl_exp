import matplotlib.pyplot as plt
import csv
import os
import copy

import numpy as np
import math



class Drawer():
# Allows to save graphs as PNG with experiment details as TXT and data as CSV files.
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.exp_name = exp_name
        self.makeDir("./experiments")
        self.makeDir(self.output_path_root)

    def savePlotPNG(self, x, y, x_label, y_label, plot_title):
    # Save plot with only one x-y curve. x and y are lists.
        plt.subplots()
        plt.plot(x,y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
    # Save plot with several curves. Curves have the same x values in a list. y_list is a list of list of y data. 
    # legend is a list of strings corresponding to each curve.
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
        plt.ylim(bottom=0)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveCSV(self, csv_title, my_list):
    # Save a list in a csv file
        path = self.output_path_root + "/" + csv_title + ".csv"
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(my_list)
    
    def saveMultiCSV(self, csv_title, list_of_lists, legend):
    # Save several lists in the same csv file. Legend is a list of strings to put in front of each list.
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


    def saveDetails(self, doc_title, env_list, algo_list, do_list = False):
    # Save the details of the running experiment 
        path = self.output_path_root + "/" + doc_title + ".txt"
        with open(path, "w") as file:
            file.write(self.exp_name + "\n\n")
            file.write("Environments: \n")
            for env in env_list:
                file.write(env.name + ": " + str(env.params) + "\n")
            file.write("\nAlgorithms: \n")
            for algo in algo_list:
                file.write(algo.name + ": " + str(algo.params) + "\n")
            if do_list:
                file.write("\nDo list: " + str(do_list) + "\n")
                file.write("Ex: ([1, 0][1, 1] means algo1 on env1; algo2 on env1 and env2")



def draw_softmax(list_action, list_q, temperature):
# Draw an action from the list_action with corresponding q-values in list_q using a softmax
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    
    chosen_index = np.random.choice(len(list_action), 1, p=list_proba)
    return list_action[chosen_index[0]]

def draw_max(list_action, list_q):
# Draws the action with the highes Q-value
    assert len(list_action) == len(list_q)
    chosen_index = np.argmax(list_q)
    return list_action[chosen_index]


def getSoftmaxProbas(list_action, list_q, temperature):
# Returns a ist of the softmax probabilities for each actions with corresponding q-values in list_q using a softmax
    assert len(list_action) == len(list_q)
    list_proba = np.zeros(len(list_q))
    for i in range(len(list_action)):
        q = list_q[i] - max(list_q)
        list_proba[i] = math.exp(q / temperature)   # compute each term
    list_proba = list_proba/(np.sum(list_proba))    # normalize
    return list_proba


def drawChiSquare(mean, variance):
# Draws a value from a chisquare(3) distribution that is linearly parametrized to ensure a given mean and variance
    basic_var = 6
    basic_mean = 3

    a = np.sqrt(variance/basic_var)
    b = mean - basic_mean*a

    assert b >= 0, "Variance and mean in drawChiSquare will lead to potential negative sampling: a = " + str(a) + " and b = " + str(b)

    return np.random.chisquare(3)*a + b      # Has a mean of 0.6, and a variance of 1/6
