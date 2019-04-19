import numpy as np

import gym
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
    
    def __init__(self,grid_x=10,grid_y=10,n_coins=10,seed=42):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.action_space = spaces.Discrete(4)# The Space object corresponding to valid actions
        self.observation_space = spaces.Discrete(grid_x*grid_y)# The Space object corresponding to valid observations
        
        self.init_state = 0
        self.done = False
        self.current_state = self.init_state
        self.n_coins = n_coins
        self.seed(seed)
        self.init_grid()

    def init_grid(self):
        probs = np.ones(self.observation_space.n)/(self.observation_space.n - 1)
        probs[self.init_state] = 0
        self.terminal_state = np.random.choice(self.observation_space.n,1,p=probs)
        
        probs[self.terminal_state] = 0
        probs/=sum(probs)
        
        self.coins_indexes = defaultdict(int)
        for i in np.random.choice(self.observation_space.n,self.n_coins,p=probs,replace=False):
            self.coins_indexes[i] = 1

    def _idx2coords(self,index):
        return index%self.grid_x,index//self.grid_x
    
    def _coords2idx(self,x,y):
        return y*self.grid_x + x
    
    def step(self, action):
        assert self.action_space.contains(action)

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
        
        if self.current_state == self.terminal_state:
            self.close()

        reward = self._get_reward()
        return self.current_state, reward, self.done, None

    def _get_reward(self):
        if self.coins_indexes[self.current_state] ==1:
            # self.coins_indexes[state] = 1 means that current position contains a coin
            # = 0 nothing or {S,T}
            # = -1 if there WAS a coin
            # if there's a coin in the current position get the coin
            self.coins_indexes[self.current_state] = -1
            # TODO calculate the reward when the Agent collects a Coin
        
        # TODO penalize the Agent for each move
        return 0
    
    def reset(self):
        self.done = False
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
    def close(self):
        self.done = True
        return
    
    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)


if __name__ == "__main__":
    env = TabularEnv()
    env.render()
    env.step(1)
    env.step(1)
    env.step(1)
    env.step(1)
    env.render()