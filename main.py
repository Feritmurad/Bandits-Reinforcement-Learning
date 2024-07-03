#!/usr/bin/python3
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

import bandit

def main():
    timesteps = int(sys.argv[1])
    b = bandit.Bandit()
    
    regret = 0.
    cumulative_regret = []
    epsilon = 0.1
    # Initialize a dictionary to store the counts and average reward for each arm
    arms_data = {i: {'count': 0, 'average_reward': 0.0} for i in range(b.num_arms())}

    for t in range(timesteps):
        # Choose an arm
        random_number = random.random()
        #print(random_number)
        if epsilon > random_number:
            # Choose an random arm
            a = random.randint(0, b.num_arms() - 1)
        else:
            # Choose the arm with the highest average reward
            optimal_arm = max(arms_data, key=lambda arm: arms_data[arm]['average_reward'])
            a = optimal_arm
            

        # Pull the arm, obtain a reward
        ret = b.trigger(a)
        regret += b.opt() - ret
        cumulative_regret.append(regret)
        
        # Learn from a and ret
        arms_data[a]['count'] += 1
        n = arms_data[a]['count']
        current_avg = arms_data[a]['average_reward']
        new_avg = ((n - 1) * current_avg + ret) / n
        arms_data[a]['average_reward'] = new_avg

    print(regret)    
    
    # Plot rewards over time
    plt.figure(figsize=(12, 5))
    plt.plot(range(timesteps), cumulative_regret, label='Cumulative Regret', color='r')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret over Time')

    plt.show()

if __name__ == '__main__':
    main()
