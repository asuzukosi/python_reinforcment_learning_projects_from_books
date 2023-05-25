# implementation of Deep deterministic policy gradient using keras

from typing import Any
import gymnasium as gym
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


# Examining the environment we would be working with
problem = "Pendulum-v1"
env = gym.make(problem, max_episode_steps=200)

# get the number of states
num_states = env.observation_space.shape[0]
print("The number of states is ", num_states)

# get the number of actions
num_actions = env.action_space.shape[0]
print("The number of actions is ", num_actions)

# get upper boound and lower bound state values
high_bound = env.observation_space.high
low_bound = env.observation_space.low

print("The upper bound is ", high_bound)
print("The lower bound is ", low_bound)


# get the upper bound and lower bound for the action values
high_action = env.action_space.high 
low_action = env.action_space.low

print("The action upper bound is ", high_action)
print("The action lower bound is ", low_action)

target_actor = None
gamma = None
target_critic = None
critic_network = None
critic_optimizer = None
actor_network = None
actor_optimizer = None
target_actor = None


# Implementaion of the Orstein Urlhembeck noise perturbation algorithm
class OUNoise:
    def __init__(self, mean, std_diviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_diviation = std_diviation
        self.dt = dt
        self.x_initial = x_initial
        
        self.reset()
    
    
    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_diviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # after performing the perturbation set the new x values to the previous x values
        self.x_prev = x # this ensures that the next noise is dependent on the current one
        # return the x values
        return x

        
        
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            # does this mean that the mean provided should be an array?
            self.x_prev = np.zeros_like(self.mean)
            

class Buffer:
    def __init__(self, capacity=100000, batch_size=32):
        self.capacity = capacity
        self.batch_size = batch_size
        
        # buffer counter tells us the amount of times record() has been called
        self.buffer_counter = 0
        
        # in this situation instead of using list of tuples,
        # we use individual nd arrays for each element in the replay memory
        self.states = np.zeros(shape=(self.capacity, num_states))
        self.actions = np.zeros(shape=(self.capacity, num_actions))
        self.rewards = np.zeros(shape=(self.capacity, 1))
        self.next_states =  np.zeros(shape=(self.capacity, num_states))
        
    
    # takes in an observation (s, a, r, s') as a tuple called obs
    def record(self, obs):
        # extract sequence
        state, action, reward, next_state = obs
        # get the index using modulus
        index = self.buffer_counter % self.capacity
        # add new observation to replay buffer
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        
        # update buffer counter
        self.buffer_counter += 1
    
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # training and updating the actor critic networks
        with tf.GradientTape() as tape:
            target_actions = tf.cast(target_actor(next_state_batch, training=True), dtype=tf.float64) # these are the actions taken by the target actor network
            y = reward_batch + gamma * tf.cast(target_critic([next_state_batch, target_actions], training=True), dtype=tf.float64)
            critic_value = tf.cast(critic_network([state_batch, action_batch], training=True), dtype=tf.float64)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            
            critic_grad = tape.gradient(critic_loss, critic_network.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grad, critic_network.trainable_variables))
            
        with tf.GradientTape() as tape:
            actions = tf.cast(actor_network(state_batch, training=True), dtype=tf.float64)
            critic_value = tf.cast(critic_network([state_batch, actions], training=True), dtype=tf.float64)
            
            actor_loss = -tf.math.reduce_mean(critic_value)
            actor_grad = tape.gradient(actor_loss, actor_network.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_network.trainable_variables))
    
    
    def learn(self):
        # get the learning_range
        learning_range = min(self.batch_size, self.buffer_counter)
        
        # get the indices of values to be trained
        indices = np.random.choice(learning_range, self.batch_size)
        
        states = tf.convert_to_tensor(self.states[indices])
        actions = tf.convert_to_tensor(self.actions[indices])
        rewards = tf.convert_to_tensor(self.rewards[indices])
        next_states = tf.convert_to_tensor(self.next_states[indices])

        self.update(states, actions, rewards, next_states)

@tf.function    
def update_targets(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
        
        
# we define functions to create our actor and critic networks
def getActor():
    # set our weights to be initialized between -3e-3 and 3e-3
    last_init  = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    inputs = tf.keras.layers.Input(shape=(num_states,))
    out = tf.keras.layers.Dense(256, activation="relu")(inputs)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    outputs = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs *= high_action
    
    model = keras.Model(inputs, outputs)
    return model



def getCritic():
    # state input
    state_input =  tf.keras.layers.Input(shape=(num_states, ))
    state_out = tf.keras.layers.Dense(16, activation="relu")(state_input)
    state_out = tf.keras.layers.Dense(32, activation="relu")(state_out)

    # action input
    action_input = tf.keras.layers.Input(shape=(num_actions, ))
    action_out = tf.keras.layers.Dense(32, activation="relu")(action_input)
    
    # add both layers together
    concat = tf.keras.layers.Concatenate()([state_out, action_out])
    
    out = tf.keras.layers.Dense(256, activation="relu")(concat)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    outputs = tf.keras.layers.Dense(1)(out)
    
    model = tf.keras.Model([state_input, action_input], outputs)
    return model


# now we implement the policy function
def policy(state, noise_object):
    sampled_actions = tf.squeeze(tf.cast(actor_network(state), dtype=tf.float64))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    legal_action = np.clip(sampled_actions, low_action, high_action)
    return [np.squeeze(legal_action)]


std_dev = 0.2
ou_noise = OUNoise(mean=np.zeros(1), std_diviation=float(std_dev) * np.ones(1))

actor_network = getActor()
critic_network = getCritic()


target_actor = getActor()
target_critic = getCritic()

target_critic.set_weights(critic_network.get_weights())
target_actor.set_weights(actor_network.get_weights())


critic_lr = 0.002
actor_lr = 0.001


critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)
actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)


total_episodes = 100
gamma = 0.99 # discount factor

tau = 0.005

buffer = Buffer(50000, 64)

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    (prev_state, _) = env.reset()
    episodic_rewards = 0
    
    while True:
        # print("Starting new time step")
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        # print("Getting action from policy")
        action = policy(tf_prev_state, ou_noise)
        
        # perform state transition
        # print("Performing state transition")
        state, reward, terminated, trunctated, info = env.step(action)
        buffer.record((prev_state, action, reward, state))
        episodic_rewards += reward
        # print("About to start learning ")
        buffer.learn()
        # print("Done learning")
        
        # print("About to update parameters")
        update_targets(target_actor.variables, actor_network.variables, tau)
        update_targets(target_critic.variables, critic_network.variables, tau)
        # print("Done updating parameters")
        
        if terminated or trunctated:
            break
        
        prev_state = state
        # print("Ending time step")/
        
    ep_reward_list.append(episodic_rewards)
    
    avg_reward = np.mean(ep_reward_list[-40:])
    print(f"Episode * {ep} * Avg Reward is => {avg_reward}")
    avg_reward_list.append(avg_reward)
    

# plotting the results
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. episode rewards")
plt.show()


actor_network.save_weights("pendulum_actor.h5")
critic_network.save_weights("pendulum_critic.h5")

target_actor.save_weights("pendulum_target_actor.h5")
target_critic.save_weights("pendulum_target_critic.h5")


# view the trained model
problem = "Pendulum-v1"
env = gym.make(problem, render_mode="human")

evaluation_episodes = 10
for episode in range(evaluation_episodes):
    (prev_state, _) = env.reset()
    
    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)
        
        # perform state transition
        state, reward, terminated, trunctated, info = env.step(action)
        
        # if it is a the terminal state update break out of the loop
        if terminated:
            break
        
        # update previous state to be the current state
        prev_state = state
        

