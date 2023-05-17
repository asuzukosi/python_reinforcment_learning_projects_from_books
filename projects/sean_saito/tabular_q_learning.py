import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import logging
import gymnasium as gym
import logging
import time
import math
from tqdm import tqdm

# allow environment to specify whether quantization is needed

# logger setup
log_format = '%(asctime)s | %(levelname)s: %(message)s'
logging.basicConfig(format=log_format, level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class EpisodeInfo():
    def __init__(self):
        pass

# class TrainingInfo():
#     pass

class ModelBuilder:
    def __init__(self, environment="cart-pole", testing_environment=None,
                 algorithm="q-learning", min_learning_rate=0.1, 
                 min_eploration=0.01, discount_factor=0.9, 
                 q_table_creation_strategy="zeros",
                 max_steps_per_episode=250, use_quantization=True, 
                 num_bins=10, q_table_path=None, num_actions=None, 
                 state_size=None, min_state=None, max_state=None, 
                 continuous=None):
        logger.info("Initializing Model Builder")
        # the algorithm to be used in this environment
        self.algorithm = algorithm
        # the minimum learning rate
        self.min_learning_rate = min_learning_rate
        # the minimum eploration 
        self.min_eploration = min_eploration
        # discount factor determines how much attention we pay to future rewards
        self.discount = discount_factor
        # specify whether the learner should use quantization or not
        self.use_quantization = use_quantization
        # specify the number of bins if quantization is to be used
        self.num_bins = num_bins
        # specify the q table path if none currently exists
        self.q_table_path=q_table_path
        # specify the q table creation strategy i.e either zero or random
        self.q_table_creation_strategy = q_table_creation_strategy
        # specify maximum number of steps per episode
        self.max_num_steps = max_steps_per_episode
        
        # get environment information, which creates properties for num actions, state size, minimum state, maximum state and q table size
        # if the environmet is a string then we will retrieve the environment and create the training and the testing environment
        if type(environment) == str:
            self.env, self.test_env = self._get_environment_from_name(environment)
            self._get_environment_info()

        else:
            # if the environment is an already existing environment then the user must provide both the training and testing environment
            if testing_environment is None or num_actions is None or state_size is None or min_state is None or max_state is None or continuous is None:
                raise ValueError("The testing_environment and other required parameters such as num_actions, state_size, min_state, continuous,"
                                 "max_state, must also be provided if using a non supported environment")
            self.env, self.test_env = environment, testing_environment # create training and testing environment
            # manually pass environment related infomation
            self.num_actions = num_actions
            self.state_size = state_size
            self.min_state = min_state
            self.max_state = max_state
            self.continous = continuous
            # get q table size
            size = self._create_bins()
            self.state_size = size
            size.append(self.num_actions)
            self.q_table_size = tuple(size)
            self._log_environment_init_values()
            
        # create q table
        self.q_table = self._get_q_table()
        self.training_episodes = None
        logger.info("Done Initializing model builder")
                
    def _get_environment_info(self):
        # get details about the environment being provided
        self.num_actions = self.env.action_space.n
        self.state_size = len(self.env.observation_space.sample()) # get size of state space from sample
        self.min_state = self.env.observation_space.low
        self.max_state = self.env.observation_space.high
        # get q table size
        size = self._create_bins()
        self.state_size = size
        size.append(self.num_actions)
        self.q_table_size = tuple(size)
        self._log_environment_init_values()
    
    def _log_environment_init_values(self):
        # show information computed
        logger.info(f"NUMBER OF ACTIONS : {self.num_actions}")
        logger.info(f"STATE SIZE : {self.state_size}")
        logger.info(f"MAX_STATE : {self.max_state}")
        logger.info(f"MIN STATE : {self.min_state}")
        logger.info(f"QTABLE SIZE : {self.q_table_size}")
      
        
        
    def _get_q_table(self, can_use_path=True):
        # creates a new q table
        if self.q_table_path is not None and can_use_path:
            q_table = np.load(self.q_table_path)
            if q_table.shape != self.q_table_size:
                raise ValueError("Q table provided in path is not compatible with the environment provided")
            else:
                return q_table
        else:
            if self.q_table_creation_strategy == "zeros":
                # if the strategy is zeros, then return a q table of zeros
                q_table = np.zeros((self.q_table_size))
            else:
                # if the strategy is random then create q table of random values
                q_table = np.random.uniform(0, 1, size=self.q_table_size)
            # return q_table
            return q_table
    
    def regenerate_q_table(self):
        # regenerates new q table and does not allow loading from path
        self.q_table = self._get_q_table(False)
        
        
    def _get_environment_from_name(self, name):
        # select environment based on name specified
        if name == "cart-pole":
            # create set of parameters for the cart-pole environment
            env = gym.make("CartPole-v1", max_episode_steps=self.max_num_steps)
            test_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=self.max_num_steps)
            self.continous = True # this specifies that the environment is continous. i.e does not have a workable terminal state
            
        elif name  == "mountain-car":
            env = gym.make('MountainCar-v0', max_episode_steps=self.max_num_steps)
            test_env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=self.max_num_steps)
            self.continous = False
            
        elif name == "lunar-lander":
            env = gym.make("LunarLander-v2", max_episode_steps=self.max_num_steps)
            test_env = gym.make("LunarLander-v2", render_mode="human", max_episode_steps=self.max_num_steps)
            self.continous = False
            
        else:
            raise ValueError(f"environment with name {name} is not supported, "
                             "please overide the _get_environment_from_name function to support new environments or pass in custom environment.")
        # return created environment
        return env, test_env
    
    def _quantize_single_state_value(self, min_value, max_value, shift=1e-10):
        gap = max_value - min_value
        interval = gap/self.num_bins
        
        start = min_value - shift
        bins = []
        bins.append(start)
        # if the numbers are too small and can not be properly split then return just the two values
        if gap == float("inf") or interval == float("inf"):
            bins.append(max_value)
        
        else:
            for i in range(self.num_bins-1):
                start += interval
                bins.append(start)
            bins.append(max_value)
            # returns a list of bins for the data
        return bins
            
    def _create_bins(self):
        state_bins = []
        # loop for each state value the minumum and the maximum
        for min_value, max_value in zip(self.min_state, self.max_state):
            bins = self._quantize_single_state_value(min_value, max_value)
            state_bins.append(bins)
        self.state_bins = state_bins
        size = []
        for i in self.state_bins:
            size.append(len(i) - 1)
        # return the state size from splitting data into bins
        return size
    
    def _quantize(self, observation):
        result = []
        for sv, bins in zip(observation, self.state_bins):
            if sv > bins[-1]:
                value = len(bins) - 2 # the last index is -2 because the bins is one 
                                      #larger than the available labels, because they work in ranges
                result.append(value)
                
            elif sv < bins[0]:
                result.append(0) # if bin is smaller than the smallest 
                                 # value then it should be set to index 0
            else:
                value = pd.cut([sv], bins=bins, labels=range(len(bins)-1))[0]
                result.append(value)
            
        return result
    # create function for selecting exploration rate
    def select_exploration_rate(self, x):
        return max(self.min_eploration, min(1, 1 - math.log10((x+1)/25)))

    # create a function for selecting learning rate
    def select_learning_rate(self, x):
        return max(self.min_learning_rate, min(0.5, 1 - math.log10((x+1)/25)))

    def _act(self, state, epsilon=0.1, training=True, random=False):
        if random:
            return np.random.randint(0, self.num_actions)
        
        # if training select action either greedily or episole greedily
        if training:
            if np.random.random() <= epsilon:
                return np.random.randint(0, self.num_actions)
            else:
                return np.argmax(self.q_table[*state])
        
        # act greedily with the q_table
        return np.argmax(self.q_table[*state])
    
    
    def _get_max_state_value(self, state):
        # the state value is the maximum available in the state
        return np.max(self.q_table[*state])
    
    def _save_q_table(self):
        # save q table in specified path and return path
        if self.q_table_path is not None:
            np.save(self.q_table_path, self.q_table)
            return self.q_table_path
        else:
            # if q_table path not specified create new q_table path and save it there
            os.makedirs("q_tables", exist_ok=True)
            self.q_table_path = os.path.join("q_tables", f"q_table_{time.time()}.npy")
            np.save(self.q_table_path, self.q_table)
            return self.q_table_path
        
    def plot_training(self):
        if self.training_episodes is None or len(self.training_episodes) == 0:
            raise ValueError("Training has not occured, can not plot graphs.")
        
        # plot total rewards
        plt.plot(range(len(self.training_episodes)), [episode[0] for episode in self.training_episodes], label="Total rewards")        
        plt.legend(loc=4)
        plt.show()
        plt.plot(range(len(self.training_episodes)), [episode[1] for episode in self.training_episodes], label="Mean rewards")
        plt.legend(loc=4)
        plt.show()
        
    def train(self, num_episodes=100, show_every=10):
        self.training_episodes = []
        # loop through all episodes
        for episode in tqdm(range(num_episodes)):
            # reset environment at the beginning of every episode
            (start_state, env_info) = self.env.reset()
            
            # set the current state to the start state
            current_state = self._quantize(start_state)
            learning_rate = self.select_learning_rate(episode)
            epsilon = self.select_exploration_rate(episode)
            rewards = []
            while True:
                # take and action based on the state
                action = self._act(current_state, training=True, epsilon=epsilon)
                # receive state dynmics information from taking a step
                observation, reward, done, truncated, info =  self.env.step(action)
                rewards.append(reward) # add reward to the mean reward
                # quantize new state from the observation
                next_state = self._quantize(observation)
                # get the state action index by appending the action to the current state (which is now the previous state)
                state_action_index = [*current_state, action]
                # use q learning algorithm to update q table
                self.q_table[*state_action_index] += learning_rate * (reward + 
                                                                      (self.discount * self._get_max_state_value(next_state)) - 
                                                                      self.q_table[*state_action_index])
                
                    
                # if the environment is continous and has truncated then break the loop
                if self.continous and truncated:
                    break
                
                # if the environment is continous and is done or has been truncated then break the loop
                if (not self.continous and done) or truncated:
                    break
                
                # set the new observation to the current state
                current_state = next_state
            
            self.training_episodes.append((np.sum(rewards), np.mean(rewards))) # add total rewards
                                                                             # and mean reward to the training episode list
            if episode % show_every == 0:
                logger.info(f'Episode : {episode}')
                logger.info(f'Mean reward : {np.mean(rewards)}')
                logger.info(f'Min reward : {np.min(rewards)}')
                logger.info(f'Max reward : {np.max(rewards)}')
                logger.info(f'Total reward : {np.sum(rewards)}')
                logger.info(f'Learning rate : {learning_rate}')
                logger.info(f'Explore rate : {epsilon}')
            # save q table at the end of every episode
            self._save_q_table()
        logger.info("Done training")
        self.plot_training()
    
    def demo(self, num_episodes=10, random=False):
        for episode in tqdm(range(num_episodes)):
            # reset environment at the beginning of every episode
            (start_state, env_info) = self.test_env.reset()
            # set the current state to the start state
            current_state = self._quantize(start_state)
            while True:
                # render the environment
                self.test_env.render()
                # take and action based on the state
                action = self._act(current_state, random=random, training=False)
                # receive state dynmics information from taking a step
                observation, reward, done, truncated, info =  self.test_env.step(action)
                
                # if the environment is continous and has truncated then break the loop
                if self.continous and truncated:
                    break
                
                # if the environment is continous and is done or has been truncated then break the loop
                if (not self.continous and done) or truncated:
                    break
                
                # set the new observation to the current state
                current_state = self._quantize(observation)
            
    
    


# Testing Model Builder thus far

# TEST 1
# create model Builder without parameters
# should create default model with cart pole and q learning algorithm
# model = ModelBuilder(environment="mountain-car")
# print(model) # works fine creates default cart pole environement with zeros q table initialization

# TEST 2
# create model builder and save q table, create q_table with random initialization strategy
# model = ModelBuilder(q_table_creation_strategy="random")
# path = model._save_q_table()
# print(path)
# data = np.load("q_tables/q_table_1684285642.7818491.npy")
# print(data) # works fine

# TEST 3
# ensure we can not create environment with invalid name
# model = ModelBuilder("random-name")
# print(model) # works fine, does not create model and shows error message

# TEST 4
# create model builder with already existing environments
# env = gym.make("CartPole-v1", max_episode_steps=250)
# test_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=250)
# max_state = env.observation_space.high
# min_state = env.observation_space.low
# num_actions = env.action_space.n
# state_size = len(env.observation_space.sample())
# model = ModelBuilder(environment=env, testing_environment=test_env, 
#                      num_actions=num_actions, state_size=state_size, 
#                      max_state=max_state, min_state=min_state)
# print(model) # works fine


# TEST 5 
# try out demo functionality
# model = ModelBuilder(environment="lunar-lander", q_table_creation_strategy="random")
# model.demo(random=False) # works fine

# TEST 6
# test appropriate creation of bins
# model = ModelBuilder()
# print(model.state_bins)

# TEST 7
# test the ability for the model builder to quatize properly
# model = ModelBuilder(q_table_creation_strategy="random")
# result = model._quantize(model.env.observation_space.sample())
# print(result)
# print(model.q_table.shape)
# print(model.q_table[*result])

# TEST 8
# try out demo functionality
# model = ModelBuilder(q_table_path="q_tables/q_table_1684305916.0623639.npy")
# model.demo(num_episodes=3) # works fine
# model.train(num_episodes=5000, show_every=10)
# model.demo(num_episodes=10)


# TEST 9
# try out demo functionality
# model = ModelBuilder(environment="mountain-car", q_table_path="q_tables/q_table_1684307953.387271.npy")
# model.demo(num_episodes=3) # works fine
# model.train(num_episodes=5000, show_every=10)
# model.demo(num_episodes=10)


# TEST 10
# try out demo functionality on moon lander
model = ModelBuilder(environment="lunar-lander", q_table_path="projects/sean_saito/tabular_q_learning.py")
model.demo(num_episodes=3) # works fine
# model.train(num_episodes=5000, show_every=10)
# model.demo(num_episodes=10)