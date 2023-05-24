# implementation of classic control problems using reinforcement learning and gymnasium
import gym
import time
import numpy as np
import tensorflow as tf

def runTask(taskName):
    try:
        env = gym.make(taskName, render_mode="human", continuous=True, max_episode_steps=150)
    except:
        env = gym.make(taskName, render_mode="human", max_episode_steps=150)
    (observation, _)  = env.reset()
    
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, trunctated, _ = env.step(action)
        print(f"Action : {action} reward : {reward} observation : {observation}")
        
        if terminated or trunctated:
            print("Game finished")
            break
        
        time.sleep(0.05)
        
    env.close()


class Task:
    def __init__(self, name):
        assert name in ["CartPole-v1", "MountainCarContinuous-v0", 
                        "Pendulum-v1", "Acrobot-v1"]
        self.name = name
        
        self.task = gym.make(name, render_mode="human")
        (self.last_state, _) = self.reset()
        
    def reset(self):
        (state, _) = self.task.reset()
        self.total_reward = 0
        return state
    
    def play_action(self, action):
        if self.name not in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            action = np.fmax(action, 0)
            action = action / np.sum(action)
            action = np.random.choice(range(len(action)), p=action)
        else:
            low = self.task.env.action_space.low
            high = self.task.env.action_space.high
            action = np.fmin(np.fmax(action, low), high)
            
        state, reward, terminated, _, _ = self.task.step(action)
        self.total_reward += 0
        return reward, state, terminated
    
    def get_total_reward(self):
        return self.total_reward
    
    def get_actoin_dim(self):
        if self.name not in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            return self.task.env.action_space.n
        else:
            return self.task.env.action_space.shape[0]
        
    def get_state_dim(self):
        return self.last_state.shape[0]
    
    def get_activation_fn(self):
        if self.name not in ["Pendulum-v1", "MountainCarContinuous-v0"]:
            return tf.nn.softmax
        else:
            return None

    

# if __name__ == "__main__":
#     taskNames = ["CartPole-v1", "MountainCarContinuous-v0", 
#                  "Pendulum-v1", "Acrobot-v1", "CarRacing-v2"]
    
#     for task in taskNames:
#         print(f"Starting task {task}")
#         runTask(task)