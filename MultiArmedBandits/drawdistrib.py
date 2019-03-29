from utils import *
import matplotlib.pyplot as plt
import csv
import os
import copy
import numpy as np
import math



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

    def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
        plt.subplots()
        for y_id in range(len(y_list)):
            y = y_list[y_id]
            y_smooth = smoothen(y)
            if legend:
                plt.plot(x,y, label = legend[y_id])
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
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


def countInDiscretizedList(number, a_list, low_lim, high_lim, coeff):
    step_size = (float(high_lim) - float(low_lim))/len(a_list)
    index = int((number-low_lim)/step_size)
    if index < len(a_list) and index >= 0:
        a_list[index] += coeff


if __name__ == "__main__":

    mean = 2

    mat = [[0 for k in range(100)], [0 for k in range(100)], [0 for k in range(100)], [0 for k in range(100)]]

    v_1_avg = 0
    v_2_avg = 0
    v_3_avg = 0

    draw_1_mem = []
    draw_2_mem = []
    draw_3_mem = []
    draw_4_mem = []

    for i in range(1000000):
        v_1 = np.random.chisquare(3)/3
        v_1_avg += (v_1 - v_1_avg)/(i+1)
        v_2 = np.random.chisquare(3)/5 + 0.4
        v_2_avg += (v_2 - v_2_avg)/(i+1)
        v_3 = np.random.chisquare(3)/10 + 0.7
        v_3_avg += (v_3 - v_3_avg)/(i+1)

        draw_1 = np.random.normal(loc = mean, scale = 1)
        draw_2 = np.random.normal(loc = mean, scale = math.sqrt(v_1))
        draw_3 = np.random.normal(loc = mean, scale = math.sqrt(v_2))
        draw_4 = np.random.normal(loc = mean, scale = math.sqrt(v_3))
        countInDiscretizedList(draw_1, mat[0], 0, 4, 1.0/1000000)
        countInDiscretizedList(draw_2, mat[1], 0, 4, 1.0/1000000)
        countInDiscretizedList(draw_3, mat[2], 0, 4, 1.0/1000000)
        countInDiscretizedList(draw_4, mat[3], 0, 4, 1.0/1000000)
        draw_1_mem.append(draw_1)
        draw_2_mem.append(draw_2)
        draw_3_mem.append(draw_3)
        draw_4_mem.append(draw_4)

    
    drawer = Drawer("TraceProbaDistros")
    x = [float(i)/100 * 4 for i in range(100)]
    drawer.saveMultiPlotPNG(x, mat, "Random variable", "Probability", "Probability distributions - Gaussians with random variance", ["N(2, 1)","N(2, X^2(3)/3)", "N(2, X^2(3)/5 + 0.4)", "N(2, X^2(3)/10 + 0.7)"])

    var_draw_1 = np.var(draw_1_mem)
    var_draw_2 = np.var(draw_2_mem)
    var_draw_3 = np.var(draw_3_mem)
    var_draw_4 = np.var(draw_4_mem)

    print("var_draw_1: " + str(var_draw_1))
    print("var_draw_2: " + str(var_draw_2))
    print("var_draw_3: " + str(var_draw_3))
    print("var_draw_4: " + str(var_draw_4))


