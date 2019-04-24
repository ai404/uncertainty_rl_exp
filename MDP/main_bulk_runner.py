from environment import *
from algorithms import *
from utils import *
import time
from main import *

def run_experiment(text_data,algo1,algo2 = None,nb_runs=300,nb_episodes=400):
    _,_,Env,mean_ter,var_ter,mean_step,var_step = text_data.strip().split(",")
    #print(text_data.strip().split(","))
    
    mean_ter = float(mean_ter) if mean_ter!="None" else None
    var_ter = float(var_ter) if var_ter!="None" else None
    mean_step = float(mean_step) if mean_step!="None" else None
    var_step = float(var_step) if var_step!="None" else None
    exp_name = "_".join(text_data.strip().split(","))

    temperature = 1
    rew_params = {"rvar_mean_ter": mean_ter, "rvar_var_ter": var_ter, "rvar_mean_step": mean_step, "rvar_var_step": var_step}
    #print(rew_params)
    if Env == "Sparse":
        env = SparseTabularEnvironment
    elif Env == "Semisparse":
        env = SemiSparseTabularEnvironment
    elif Env == "Dense":
        env = DenseTabularEnvironment
        temperature = 10
    
    instance_env = env(6, 6, rew_params)
    #print(instance_env.action_space.n)
    algo_params1 = {"action_space": instance_env.action_space, "temperature": temperature, "alpha": 0.3, "gamma": 1}
    if algo2 is None:
        instance_algo1 = algo1(algo_params1)
        
        trainer1 = Trainer(instance_env, instance_algo1)
        train_returns1 = trainer1.evalAvgReturn([nb_runs, nb_episodes])

        # # Plotting
        drawer = Drawer(exp_name)

        legend =  [str(rew_params)]
        drawer.saveMultiPlotPNG(range(len(train_returns1)), [train_returns1], "Episode", "Average return", instance_env.getName() + ": return averaged on " + str(nb_runs) + " runs using " + instance_algo1.getName(), legend)
        drawer.saveMultiCSV(instance_env.getName() + ": return averaged on " + str(nb_runs) + " runs using " + instance_algo1.getName(), [train_returns1], legend)
        
    else:
        instance_algo1 = algo1(algo_params1)
        instance_algo2 = algo2(algo_params1)
        compare([instance_algo1, instance_algo2], [instance_env], exp_name, nb_runs, nb_episodes)

if __name__ == '__main__':
    
    # Experiment parameters
    nb_runs = 300
    nb_episodes = 400

    with open("experiments.csv","r") as f:
        for i,line in enumerate(f.readlines()):
            if i==0:continue
            run_experiment(line,Sarsa,nb_runs=nb_runs,nb_episodes=nb_episodes)
            break
            