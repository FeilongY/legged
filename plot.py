import numpy as np
import matplotlib.pyplot as plt
import os

#class PlotReward:

def __init__(self,
                log_dir=None,
                ):
    # Log
    self.log_dir = log_dir
    
def plot(runname):
    reward = []
    # open file and read the content in a list
    with open(os.path.join('/home/feilong/Desktop/isaacgym/legged_gym/logs/rough_a1',runname, 'reward.txt'), "r") as fp:
        line = fp.read()
            # remove linebreak from a current name
            # linebreak is the last character of each line
            # x = line[:-1]

            # add current item to the list
        reward.append(line)

    # display list
    print(reward)

plot('Sep09_04-16-21_stairup')


