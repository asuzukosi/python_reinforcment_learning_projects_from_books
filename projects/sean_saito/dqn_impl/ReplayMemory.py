import numpy as np
import gymnasium as gym
from PIL import Image
from collections import deque
import warnings
import random
import cv2

def convert_to_grayscale(observation):
    # this method converts an image received from the environment into a grayscale image
    observation = np.array(observation)
    observation = np.mean(observation, axis=2)
    return observation

def normalize_image(observation):
    # scale down the observation so it will be easier for the neural network to process
    observation = np.array(observation) / 255
    return observation

def resize_image(im, resize_shape=(84, 84), methods="crop", crop_offset=8):
    # im is a PIL Image class
    height, width = im.shape
    resize_height, resize_width = resize_shape
    
    if methods == "crop":
        h = int(round(float(height) * resize_height / width))
        resized = cv2.resize(im, (resize_width, h), interpolation=cv2.INTER_LINEAR)
        crop_y_cutoff = h - crop_offset - resize_height
        cropped = resized[crop_y_cutoff:crop_y_cutoff + resize_height, :]
        return np.asarray(cropped, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown image resize method {methods}")

def crop_image_to_84_84(observation):
    # perform image scaling here
    observation = resize_image(observation)
    return observation

class ReplayMemory:
    # generalizable replay memory class which can be used to store any type of data
    def __init__(self, capacity=1000, history_len=4, num_null_operations=4, batch_size=32, transformations=None):
        # specify the transformations that should be done on the state 
        # observation before it is saved to the replay memory
        self.capacity = capacity
        self.history_len = history_len
        self.batch_size = batch_size
        self.transformations = transformations
        if num_null_operations  < self.history_len:
            raise ValueError("Number of null operations can not be less than the history length")
        self.num_null_operations = num_null_operations
        
        # create two parallel lists to store the observation and other experience details
        self.states = deque([])
        self.others = deque([])
    
    # get the current size fo the replay memory
    def get_current_size(self):
        assert len(self.states) == len(self.others)
        return len(self.states)
    
    def apply_transformations(self, observation):
        if observation is None:
            raise ValueError("Observation is None")
        # loop through all transformations
        for i in range(len(self.transformations)):
            # apply all transformations on the observation
            observation = self.transformations[i](observation)
            # if a transformation returns a non objece raise a value error
            if observation is None:
                raise ValueError(f"Transformation {i} has transformed the observation to a None object")
        return observation
            
    def add(self, observation, action, reward, termination):
        # if the replay memory is full then remove the earliest item
        if len(self.states) >= self.capacity:
            self.states.popleft()
            self.others.popleft()
        
        # ensure that the length of states and others are the same size
        assert len(self.states) == len(self.others)
        # apply transformations on the observation
        observation = self.apply_transformations(observation)
        
        # add observation to the replay memory
        self.states.append(observation)
        # add other experiences to the replay memory
        self.others.append((action, reward, termination))
        
    def add_null_operations(self, init_observation):
        # we add these null operations to serve as a padding for our
        # replay memory in situations where the agent is trying to take its first actions and
        # the replay memory is not full enough to construct a full state
        for _ in range(self.num_null_operations):
            # add null experiences to the replay memory
            self.add(init_observation, 0, 0, 0)
    
    # the states as stored in the replay memory are in single form, to turn the state into a continous
    # sequence we will create a 'full state' which is a sequence of successive states
    def get_full_state_from_observation(self, observation, transform=True):
        if transform:
            observation = self.apply_transformations(observation)
        # chec if the there are enough states to create a full state
        if len(self.states) < self.history_len:
            warnings.warn("Replay memory does not have enough states to construct a full state,"
                          " would be creating null sstates with this state")
            
            self.add_null_operations(observation)
        full_state = list(self.states)[-self.history_len + 1:] + [observation] # get the last history - 1 states plus this new observation
        return np.array(full_state)
            

    def get_full_state_from_index(self, index):
        if index >= len(self.states) -2 or index < self.history_len-1:
            raise ValueError("Index out of range")
        full_state = list(self.states)[index - self.history_len + 1: index + 1] # get the state sequence starting till index of size history len
        return np.array(full_state)
    
    def _sequence_has_terminal_state(self, index):
        infos = list(self.others)[index - self.history_len + 1: index + 1] # get the state sequence starting till index of size history len
        for i in range(0, len(infos)- 1):
            # this means that this sequence has a terminal state
            if infos[i][2] == True:
                return True
        return False
    
    def sample(self):
        # randomly choose an index that does not have a terminal state
        index = None
        while True:
            index = random.randint(a=(self.history_len-1), b=len(self.states)-2)
            if not self._sequence_has_terminal_state(index):
                break
        state = self.get_full_state_from_index(index)
        new_state = self.get_full_state_from_index(index + 1)
        action, reward, terminated = self.others[index]
        
        return (state, action, reward, new_state, terminated)
        
    
memory = ReplayMemory(transformations=[convert_to_grayscale, normalize_image, crop_image_to_84_84])

env =  gym.make("Breakout-v4")
(observation, info) = env.reset()
memory.add_null_operations(observation)

for i in range(20):
    observation = memory.get_full_state_from_observation(observation)
    
    print("Observation constructed is of shape: ", observation.shape)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    memory.add(observation, action, reward, terminated)
    
states = memory.states
print(np.array(states).shape)

state = memory.get_full_state_from_index(10)
print("Generated state is of shape: ", state.shape)

sampled_state = memory.sample()
print("Sampled state is of shape: ", sampled_state[0].shape)