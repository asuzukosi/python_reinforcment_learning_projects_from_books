import logging
import gym
import numpy as np
import tensorflow as tf

from projects.sean_saito.utils import cv2_resize_image
# import minecraft_py

# logging.basicConfig(level=logging.DEBUG)
# proc, _ = minecraft_py.start()
# minecraft_py.stop(proc)

class Game:
    def __init__(self, name="MinecraftBasic-v0", discrete_movement=False):
        self.env = gym.make(name)
        if discrete_movement:
            self.env.init(start_minecraft=True, allowDiscreteMovement=["move", "turn"])
        else:
            self.env.init(start_minecraft=True, allowContinuousMovement=["move", "turn"])
        
        self.action = list(range(self.env.action_space.n))
        frame = self.env.reset()
        
        self.frame_skip = 4
        self.total_reward = 0
        self.crop_size = 84
        self.buffer_size = 8
        
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame
        
    def rgb_to_gray(self, im):
        return np.dot(im, [0.2126, 0.7152, 0.0722])
    
    def reset(self):
        frame = self.env.reset()
        self.total_reward = 0
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_index)]
        self.last_frame = frame
        
    def add_frame_to_buffer(self, frame):
        self.buffer_index = self.buffer_index % self.buffer_size
        self.buffer[self.buffer_index] = frame
        self.buffer_index += 1
        
    def get_available_actions(self):
        return list(range(len(self.actions)))
    
    def get_feeedback_size(self):
        return (self.crop_size, self.crop_size)
    
    def crop(self, frame):
        feedback = cv2_resize_image(frame, resized_shape=(self.crop_size, self.crop_size), 
                                    method="scale", crop_offset=0)
        return feedback
    
    def get_current_feedback(self, num_frames=4):
        assert num_frames < self.buffer_size, "Frame buffer is not large enough"
        index = self.buffer_index - 1
        frames = [np.expand_dims(self.buffer[index - k], axis=0) for k in range(num_frames)]
        
        if num_frames > 1:
            return np.concatenate(frames, axis=0)
        else:
            return frames[0]
        
    def play_action(self, action, num_frames=4):
        reward = 0
        termination = 0
        for i in range(self.frame_skip):
            a = self.action[action]
            frame, r, terminated, truncated, _ = self.env.step(action)
            reward += r
            if i == self.frame_skip -2:
                self.last_frame = frame
            if terminated:
                termination = 1
        self.add_frame_to_buffer(self.crop(np.maximum(self.rgb_to_gray(frame), 
                                                      self.rgb_to_gray(terminated), self.rgb_to_gray(self.last_frame))))
        
        r = np.clip(reward, -1, 1)
        self.total_reward += reward
        return r, self.get_current_feedback(num_frames), termination
    
    
    
class FFPolicy:
    def __init__(self, input_shape=(84, 84, 4), n_outputs=4, network_type="cnn"):
        self.width, self.height, self.channels = input_shape
        self.n_outputs = n_outputs
        self.network_type = network_type
        self.entropy_beta = 0.01
        
        
    def build_model(self):
        # create sequential model
        self.net = tf.keras.models.Sequential()
        
        if self.network_type == "cnn":
            self.net.add(tf.keras.layers.Conv2D(filters=16, 
                                                kernel_size=(8, 8), 
                                                strides=(4, 4),
                                                input_shape=(84, 84, 4), 
                                                name="conv1"))
            self.net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv2"))
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(256, name="fc1"))
        else:
            self.net.add(tf.keras.layers.Dense(50, 
                                               activation="relu", 
                                               name="fc1"))
            self.net.add(tf.keras.layers.Dense(50, name="fc2"))
            
        self.net.add(tf.keras.layers.Dense(self.n_outputs, name="value"))
        self.net.add(tf.keras.layers.Softmax(name="policy"))
        
        print(self.net.summary())
        return self.net
    
    
    
class A3C:
    def __init__(self, system, directory, param, agent_index=0, callback=None):
        self.system = system
        self.actions = system.get_available_actions()
        self.directory = directory
        self.callback = callback
        self.feedback_size = system.get_feedback_size()
        self.agent_index = agent_index
        
        self.set_params(param)
        self.init_network()
        
    def init_network(self):
        pass
        
            
            
            
        
        
        
            