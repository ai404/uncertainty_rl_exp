import numpy as np

import gym
from utils import *
from gym import spaces
from collections import defaultdict

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
class bcolors:
    START = '\033[95m'
    #BLUE = '\033[94m'
    TERMINAL = '\033[92m'
    COIN = '\033[93m'
    AGENT = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TabularEnv(gym.Env):
    metadata = {'render.modes': ['human','ansi']}
    
    def __init__(self,grid_x=5,grid_y=5,n_coins=0,seed=42, max_steps = 10000):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.action_space = spaces.Discrete(4)# The Space object corresponding to valid actions
        self.observation_space = spaces.Discrete(grid_x*grid_y)# The Space object corresponding to valid observations

        self.init_state = np.random.choice(range(grid_x*grid_y))
        self.current_state = self.init_state

        self.max_steps = max_steps
        self.step_count = 0

        self.done = False
        self.n_coins = n_coins
        self.seed(seed)
        self.init_grid()

    def init_grid(self):
        probs = np.ones(self.observation_space.n)/(self.observation_space.n - 1)
        probs[self.init_state] = 0
        self.terminal_state = self.init_state # Will be changed just afterwards
        while self.terminal_state == self.init_state:  # Making sure terminal state is not initial state
            self.terminal_state =  np.random.choice([0, self.grid_x -1, (self.grid_y-1)*self.grid_x, self.grid_x*self.grid_y - 1]) # 4 corners
        
        probs[self.terminal_state] = 0
        probs/=sum(probs)
        
        self.coins_indexes = defaultdict(int)
        for i in np.random.choice(self.observation_space.n,self.n_coins,p=probs,replace=False):
            self.coins_indexes[i] = 1

    def _idx2coords(self,index):
        return index%self.grid_x,index//self.grid_x
    
    def _coords2idx(self,x,y):
        return y*self.grid_x + x

    def getCurrentState(self):
        return self.current_state * 10**(np.ceil(np.log10(self.grid_x*self.grid_y))) + self.terminal_state

    def step(self, action):
        assert self.action_space.contains(action)
        self.step_count += 1

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
        
        reward, reward_noise, reward_var = self._get_reward()

        if self.current_state == self.terminal_state:
            #print("Reached terminal_state in " + str(self.step_count))
            reward, reward_noise, reward_var = self.close("term")

        if self.step_count >= self.max_steps:
            #print("Reached maximal steps number")
            reward, reward_noise, reward_var = self.close("max_steps")

        return self.getCurrentState(), reward, reward_noise, reward_var, self.done

    def _get_reward(self):
        # Reward, reward noise, reward noise var
        return 0, 0, 0
    
    def reset(self):
        self.done = False
        self.step_count = 0
        self.current_state = self.init_state
        self.init_grid()

    def render(self, mode='ansi'):
        ### Characters meaning:
        # A : Agent
        # S : Initial State
        # T : Terminal State
        # C : a Coin
        # X : There was a coin in that case
        # . : Nothing :3
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
                    elif self.coins_indexes[s] == 1:
                        print(bcolors.COIN+" C "+bcolors.ENDC,end="")
                    elif self.coins_indexes[s] == -1:
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


class SparseTabularEnvironment(TabularEnv):
    """docstring for SparseTabularEnvironment"""
    def __init__(self, grid_x=5,grid_y=5, reward_var_mean = None, reward_var_var = None):
        super().__init__(grid_x, grid_y)
        if reward_var_mean:
            assert reward_var_var, "You need to define variance of reward variance if you define its mean"
        self.reward_var_mean = reward_var_mean
        self.reward_var_var = reward_var_var
    

    def _get_reward(self, closing_reason = False):
        if closing_reason == "max_steps":
            return -100000, 0, 0

        elif self.current_state == self.terminal_state:
            reward_mean = 1000 + self.grid_x*self.grid_y - self.step_count    # size of grid - number of steps
            
            if self.reward_var_mean and self.reward_var_var:
                reward_var = drawChiSquare(self.reward_var_mean, self.reward_var_var)
                reward_noise = np.random.normal(loc = 0, scale = np.sqrt(reward_var))
            else:
                reward_var = None
                reward = reward_mean

            return reward_mean, reward_noise, reward_var
        else:
            return 0, 0, 1


if __name__ == "__main__":
    env = TabularEnv()
    env.render()
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.render()