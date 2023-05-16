# import required libraries
import os
import gymnasium as gym
import numpy as np
import random
import math


env = gym.make("CartPole-v1")

action_count = env.action_space.n
num_buckets = (1, 1, 6, 3)
state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
# we adjust the bounds for the second and fourth observation values
state_value_bounds[1] = [-0.5, 0.5]
state_value_bounds[3] = [-math.radians(50), math.radians(50)]

action_index = len(num_buckets)
# instantiate the q table
q_value_table = np.zeros(num_buckets + (action_count, ) )

# define minimum exploration
min_exploration = 0.01
# define minimum learning rate
min_learning_rate = 0.1

# I have absolutely no idea what any of these mean
max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0

# method used for selecting actions
def select_action(state, exploration_rate):
    # if the random value is below or equal to 
    # the exploration rate pick a random action
    if random.random() <= exploration_rate:
        action = env.action_space.sample()
    else:
        # select the action with the highest value in the q_table
        action = np.argmax(q_value_table[state])
    return action

# create function for selecting exploration rate
def select_exploration_rate(x):
    return max(min_exploration, min(1, 1 - math.log10((x+1)/25)))

# create a function for selecting learning rate
def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1 - math.log10((x+1)/25)))


# fix continous state into bucket
def bucket_state_value(state_value):
    bucket_indexes = []
    # in the case of our cart pole problem, this would be 4 because our state is a list with 4 continous values
    for i in range(len(state_value)):
        # if the value is smaller than the minimum
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        # if the value is larger than the maximum
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = num_buckets[i] -1
        # if the value is within range calculate the index it belongs to
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (num_buckets[i] - 1) * state_value_bounds[i][0]/bound_width
            scaling = (num_buckets[i] - 1)/bound_width
            bucket_index = int(round(scaling*state_value[i] - offset))
            
        # add bucket index to list of bucket indexes which will form the  path to the state
        bucket_indexes.append(bucket_index)
    
    # convert bucket index into tuple
    return tuple(bucket_indexes)
    
    
for episode in range(max_episodes):
    exploration_rate = select_exploration_rate(episode)
    learning_rate = select_exploration_rate(episode)
    
    observation = env.reset()
    start_state = bucket_state_value(observation)
    current_state = start_state
    
    for time in range(max_time_steps):
        env.render()
        selected_action = select_action(current_state, exploration_rate)
        obs, reward, terminated, truncated, info = env.step(selected_action)
        # state_value = q_value_table[previous_state + (selected_action, )]
        next_state = bucket_state_value(obs)
        next_state_best_q_value = np.max(q_value_table[next_state])
        q_value_table[current_state + (selected_action, )] += learning_rate * (reward + (discount * next_state_best_q_value) - 
                                                                               q_value_table[current_state + (selected_action, )])
        
        print('Episode number : %d' % episode)
        print('Time step : %d' % time)
        print('Selection action : %d' % selected_action)
        print('Current state : %s' % str(next_state))
        print('Reward obtained : %f' % reward)
        print('Best Q value : %f' % next_state_best_q_value)
        print('Learning rate : %f' % learning_rate)
        print('Explore rate : %f' % exploration_rate)
        print('Streak number : %d' % no_streaks)
        
        if terminated:
            print(f"episode reached terminal state after {time} time steps")
            
            if time >= solved_time:
                no_streaks += 1
            else:
                no_streaks = 0
            
        current_state = next_state
    if no_streaks > streak_to_end:
        break
        
        
        
    


