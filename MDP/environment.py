import numpy as np

import gym
from utils import *
from gym import spaces
from collections import defaultdict

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

STATE_REWARDS = {}


class bcolors:
    START = '\033[95m'
    #BLUE = '\033[94m'
    TERMINAL = '\033[92m'
    AGENT = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class breward:
    TERM = 100
    MAX_STEPS = -1000
    STEP = -1
    DENSE_RANGE = 11 # (reward going from 0 to DENSE_RANGE - 1)

class TabularEnv(gym.Env):
    metadata = {'render.modes': ['human','ansi']}
    
    def __init__(self, grid_x = 5, grid_y = 5, seed = 42, max_steps = 200):
        # Action and observation spaces
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.action_space = spaces.Discrete(4)# The Space object corresponding to valid actions
        self.observation_space = spaces.Discrete(grid_x*grid_y)# The Space object corresponding to valid observations

        # Initialising state
        self.init_state = np.random.choice(range(grid_x*grid_y))
        self.current_state = self.init_state

        # Step counting
        self.max_steps = max_steps
        self.step_count = 0

        # Others
        self.done = False
        self.seed(seed)
        self.name = "Default"

        # Initialising grid
        self.init_grid()

    def init_grid(self):
        self.terminal_state = 0
        # Indicator of state passed
        self.passed = [0 for i in range(self.grid_x*self.grid_y)]
        self.passed[self.current_state] = 1
        
    def _idx2coords(self,index):
        return index%self.grid_x,index//self.grid_x
    
    def _coords2idx(self,x,y):
        return y*self.grid_x + x

    def getCurrentState(self):
        # State given to the agent: contains both current position and terminal state
        return self.current_state * 10**(np.ceil(np.log10(self.grid_x*self.grid_y))) + self.terminal_state

    def step(self, action):
        # Check that action is legal
        assert self.action_space.contains(action)

        # Increment step count
        self.step_count += 1

        # Do action
        x,y = self._idx2coords(self.current_state)

        if action == LEFT:
            x = max(x-1,0)
        elif action == RIGHT:
            x = min(x+1,self.grid_x-1)
        elif action == UP:
            y = min(y+1,self.grid_y-1)
        elif action == DOWN:
            y = max(y-1,0)

        self.current_state = self._coords2idx(x,y)

        # Reward (depends if episode is terminated and if so, how)
        if self.current_state == self.terminal_state:
            reward, reward_noise, reward_var = self.close("term")
        elif self.step_count >= self.max_steps:
            reward, reward_noise, reward_var = self.close("max_steps")
        else:
            reward, reward_noise, reward_var = self._get_reward()

        # Marks that state has been passed before
        if self.passed[self.current_state] == 0:
            self.passed[self.current_state] = 1

        return self.getCurrentState(), reward, reward_noise, reward_var, self.done

    def _get_reward(self):
        # Reward, reward noise, reward noise var
        return 0, 0, 0
    
    def reset(self):
        self.done = False
        self.step_count = 0
        self.init_state = np.random.choice(range(self.grid_x*self.grid_y))
        self.current_state = self.init_state
        self.init_grid()

    def render(self, mode='ansi'):
        ### Characters meaning:
        # A : Agent
        # S : Initial State
        # T : Terminal State
        # X : Passed state (in this episode)
        # . : Unpassed state (in this episode)
        ### modes
        # human : for a graphical interface // not implemented
        # ansi  : console based output

        assert mode in self.metadata["render.modes"]

        if mode  == "human":
            pass
        elif mode =="ansi":
            for y in range(self.grid_y):
                print("|",end="")
                for x in range(self.grid_x):
                    s = self._coords2idx(x,y)
                    if self.current_state == s:
                        print(bcolors.AGENT+" A "+bcolors.ENDC,end="")
                    elif self.init_state == s:
                        print(bcolors.BOLD+bcolors.START+" S "+bcolors.ENDC,end="")
                    elif self.terminal_state == s:
                        print(bcolors.BOLD+bcolors.TERMINAL+" T "+bcolors.ENDC,end="")
                    elif self.passed[s] == 1:
                        print(" X ",end="")
                    else:
                        print(" . ",end="")
                print("|")
        print()

    def close(self, reason):
        self.done = True
        reward, reward_noise, reward_var = self._get_reward(reason)
        return reward, reward_noise, reward_var
    
    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def getName(self):
        return self.name

    def computeReward(self, reward_mean, reward_var_mean = None, reward_var_var = None):
        # Computes the reward, the noise and the variance given the reward mean, the expected reward variance and the variance of the reward variance.
        curr_reward_var_mean = reward_var_mean
        curr_reward_var_var = reward_var_var
        if curr_reward_var_mean and curr_reward_var_var:
            reward_var = drawChiSquare(curr_reward_var_mean, curr_reward_var_var)
            reward_noise = np.random.normal(loc = 0, scale = np.sqrt(reward_var))
        elif curr_reward_var_mean and not curr_reward_var_var:
            reward_var = curr_reward_var_mean
            reward_noise = np.random.normal(loc = 0, scale = np.sqrt(reward_var))
        else:
            reward_var = None
            reward_noise = 0
               
        return reward_mean, reward_noise, reward_var


class SparseTabularEnvironment(TabularEnv):
    """This environment gives a final reward of (1000 - number of steps that were taken to arrive there), with noise.
        Needs as reward parameters:
            reward_var_mean_ter: mean variance of the reward for the terminal state
            reward_var_var_ter: variance of the variance of the reward for the terminal state"""
    def __init__(self, grid_x, grid_y, reward_params):
        super().__init__(grid_x, grid_y)
        self.params = reward_params
        self.reward_var_mean_ter = reward_params["reward_var_mean_ter"]
        self.reward_var_var_ter = reward_params["reward_var_var_ter"]
        self.reward_var_mean_max = reward_params["reward_var_mean_max"]
        self.reward_var_var_max = reward_params["reward_var_var_max"]
        self.name = "Sparse"  

    def _get_reward(self, closing_reason = False):
        # Max steps
        if closing_reason == "max_steps":
            #print("Closing because max steps")
            return self.computeReward(breward.MAX_STEPS, self.reward_var_mean_max, self.reward_var_var_max)

        # Terminal state
        elif closing_reason == "term":
            #print("Closing because terminal state")
            reward_mean = breward.TERM + self.step_count*breward.STEP # End
            return self.computeReward(reward_mean, self.reward_var_mean_ter, self.reward_var_var_ter)

        else:
            return self.computeReward(0)


class SemiSparseTabularEnvironment(TabularEnv):
    """This environment gives a reward of 1000, with noise, at terminal state and a reward of -1, with noise, at each other state.
        Needs as reward parameters:
            reward_var_mean_ter: mean variance of the reward for the terminal state
            reward_var_var_ter: variance of the variance of the reward for the terminal state
            reward_var_mean_step: mean variance of the reward at each step
            reward_var_var_step: variance of the variance of the reward at each step"""    
    def __init__(self, grid_x, grid_y, reward_params):
        super().__init__(grid_x, grid_y)
        self.params = reward_params
        self.reward_var_mean_ter = reward_params["reward_var_mean_ter"]
        self.reward_var_var_ter = reward_params["reward_var_var_ter"]
        self.reward_var_mean_step = reward_params["reward_var_mean_step"]
        self.reward_var_var_step = reward_params["reward_var_var_step"] 
        self.reward_var_mean_max = reward_params["reward_var_mean_max"]
        self.reward_var_var_max = reward_params["reward_var_var_max"]         
        self.name = "Semi sparse"  

    def _get_reward(self, closing_reason = False):
        # Max steps
        if closing_reason == "max_steps":
            return self.computeReward(breward.MAX_STEPS, self.reward_var_mean_max, self.reward_var_var_max)

        # Terminal state
        elif closing_reason == "term":
            return self.computeReward(breward.TERM , self.reward_var_mean_ter, self.reward_var_var_ter)

        # Normal step
        else:
            return self.computeReward(breward.STEP, self.reward_var_mean_step, self.reward_var_var_step)

class DenseTabularEnvironment(TabularEnv):
    """This environment gives a reward of 1000 at terminal state and a reward of -1 at each other state.
        Additionnally, the environment gives a reward between 0 and +10 at each state if it is the first time the agent passes there.
        Needs as reward parameters:
            reward_var_mean_ter: mean variance of the reward for the terminal state
            reward_var_var_ter: variance of the variance of the reward for the terminal state
            reward_var_mean_step: mean variance of the reward at each step
            reward_var_var_step: variance of the variance of the reward at each step"""    
    def __init__(self, grid_x, grid_y, reward_params):
        super().__init__(grid_x, grid_y)
        self.params = reward_params
        self.reward_var_mean_ter = reward_params["reward_var_mean_ter"]
        self.reward_var_var_ter = reward_params["reward_var_var_ter"]
        self.reward_var_mean_step = reward_params["reward_var_mean_step"]
        self.reward_var_var_step = reward_params["reward_var_var_step"]
        self.reward_var_mean_max = reward_params["reward_var_mean_max"]
        self.reward_var_var_max = reward_params["reward_var_var_max"]
        self.name = "Dense"

        if grid_x*grid_y not in STATE_REWARDS:
            STATE_REWARDS[grid_x*grid_y] = [np.random.choice(range(breward.DENSE_RANGE)) for i in range(self.grid_x*self.grid_y)]

        self.state_reward = STATE_REWARDS[grid_x*grid_y]
        self.state_reward[self.terminal_state] = breward.TERM


    def init_grid(self):
        # Terminal state is drawn as one corner (different from first current_state)
        self.terminal_state = self.current_state # Will be changed just afterwards
        while self.terminal_state == self.current_state:  # Making sure terminal state is not initial state
            self.terminal_state =  np.random.choice([0, self.grid_x -1, (self.grid_y-1)*self.grid_x, self.grid_x*self.grid_y - 1]) # 4 corners

        # Indicator of state passed
        self.passed = [0 for i in range(self.grid_x*self.grid_y)]
        self.passed[self.current_state] = 1


    def _get_reward(self, closing_reason = False):
        # Max steps
        if closing_reason == "max_steps":
            return self.computeReward(breward.MAX_STEPS, self.reward_var_mean_max, self.reward_var_var_max)

        else:
            reward_mean = (1-self.passed[self.current_state])*self.state_reward[self.current_state] + breward.STEP
            return self.computeReward(reward_mean, self.reward_var_mean_step, self.reward_var_var_step)



if __name__ == "__main__":
  # Reward parameters
    rew_var_mean_ter = 100000
    rew_var_var_ter = 1
    rew_var_mean_step = 100
    rew_var_var_step = 1
    rew_params = {"reward_var_mean_ter": rew_var_mean_ter, "reward_var_var_ter": rew_var_var_ter, "reward_var_mean_step": rew_var_mean_step, "reward_var_var_step": rew_var_var_step}
    
    # Environment
    env = DenseTabularEnvironment(6, 6, rew_params)
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(1)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()
    
    state, reward, reward_noise, reward_var, done = env.step(1)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(2)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(2)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(3)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(0)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()

    state, reward, reward_noise, reward_var, done = env.step(0)
    print("Reward: " + str(reward) + ", noise: " + str(reward_noise) + ", var: " + str(reward_var))
    env.render()
