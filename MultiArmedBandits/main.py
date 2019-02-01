from environment import ArticleEnvironment, BookEnvironment
from algorithms import ActionEliminationAlgo, UCBAlgo, LUCBAlgo
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
        reward = self.env.draw(action_record[0])
        self.algo.update(action_record[0],reward)

        if action_record[1]:
            self.action_memory.append(action_record[0])
            self.reward_memory.append(reward)
            self.step += 1


    def findBestArm(self):
        # HERE: Choose the stop condition: algorithm has solved problem OR fixed number of steps (better to do averages per step)
        #while not self.algo.isDone():
        while self.step < self.nb_steps:
            self.doOneStep()
        return self.algo.result()

    def get_action_memory(self):
        return self.action_memory


class Drawer():
    def __init__(self, exp_name, environment):
        self.output_path_root = "./experiments/" + exp_name
        self.make_dir("./experiments")
        self.make_dir(self.output_path_root)
        self.H1 = environment.getH1()
        self.nb_arms = environment.getNbArms()

    def save_png(self, x, sum_action_step, x_label, y_label, plot_title):
        plt.subplots()
        x_normalized = [step/self.H1 for step in x]
        for i in range(self.nb_arms):
            y = sum_action_step[i]
            y = smoothen(y)
            plt.plot(x_normalized, y)
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
        if i % (nb_runs/500) == 0:
            print(i * 100 / nb_runs)
    #print(str(sum_action_step)) 

    # Drawing the results
    drawer = Drawer(exp_name, environment)
    drawer.save_png(range(nb_steps), sum_action_step, "Number of pulls (normalized by H1)", "P(I_t = i)", exp_name)
    drawer.save_csv(sum_action_step, exp_name)




if __name__ == "__main__":

    NB_RUNS = 10
    FIXED_NB_H1 = 80
    EXP_NAME = "Try"

    DELTA = 0.1
    EPSILON = 0.01


    ENVIRONMENT = BookEnvironment()
    ALGO = ActionEliminationAlgo(DELTA, EPSILON, ENVIRONMENT.getOmega())

    comparePickingProbabilities(FIXED_NB_H1, NB_RUNS, EXP_NAME, ENVIRONMENT, ALGO)