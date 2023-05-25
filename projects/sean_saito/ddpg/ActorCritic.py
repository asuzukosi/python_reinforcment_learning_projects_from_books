import tensorflow as tf
import numpy as np
import keras
from ReplayMemory import ReplayMemory


class ActorCritic:
    def __init__(self, num_actions, max_action,
                 min_action, num_states, num_actor_layers, 
                 num_critic_layers, tau, discount_rate, 
                 actor_lr, critic_lr,
                 replay_memory: ReplayMemory=None, 
                 noise_fn=None):
        # the number of actiosn to be taken at each time step
        self.num_actions = num_actions
        # get the maximum possible action in that environment
        self.max_action = max_action
        # get the minimum possible action in that environment
        self.min_action = min_action
        # the num of values to be provided as the state in each time step
        self.num_states = num_states
        # the number of layers for the actor network
        self.num_actor_layers = num_actor_layers
        # the number of layers for the critic network
        self.num_critic_layers = num_critic_layers
        # create the behaviour and the target actor network i.e b_actor and t_actor respectively
        self.b_actor = self._create_actor()
        self.t_actor = self._create_actor()
        # transfer the weights of the behaivioral actor to the target network
        self.t_actor.set_weights(self.b_actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)
        
        # create the behaviour and the target critic networks i.e b_critic and t_critic
        self.b_critic = self._create_critic()
        self.t_critic = self._create_critic()
        # transfer the weights of the behaivior critic network to the target network
        self.t_critic.set_weights(self.b_critic.get_weights())
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)
        
        # this is the tau constant used to update the trainable parameters
        self.tau = tau # tau is a value between zero and one
        self.replay_memory = replay_memory
        self.noise_fn = noise_fn
        self.discount_rate = discount_rate
        
            
    def policy(self, state):
        # Implement the policy function of the network
        state = np.array([state])
        action =  self.b_actor(state)
        noise = self.noise_fn()
        action = action.numpy() + noise # add noise to the action to be taken
        # clip action so that it would only be within the allowed range
        action = np.clip(action, self.min_action, self.max_action)
        return [np.squeeze(action)]
    
    def load_weights_from_file(self, behaviour_actor_weights_path, behaviour_critic_weights_path, 
                               target_actor_weights_path, target_critic_weights_path):
        # load weights from file paths
        self.b_actor.load_weights(behaviour_actor_weights_path)
        self.t_actor.load_weights(target_actor_weights_path)
        self.b_critic.load_weights(behaviour_critic_weights_path)
        self.t_critic.load_weights(target_critic_weights_path)
    
    def train(self):
        states, actions, rewards, next_states, terminated = self.replay_memory.sample()
        
        # update the behavioral critic network with the gradient of the loss against the trainable parameters
        with tf.GradientTape() as tape:
            next_actions = self.t_actor(next_states, training=True)
            next_values = self.t_critic([next_states, next_actions], training=True)
            y = rewards + self.discount_rate * next_values
            predictions = self.b_critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.square(y - predictions))
            critic_grad = tape.gradient(critic_loss, self.b_critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.b_critic.trainable_variables))
            
        # update the behavior actor network with the gradiets of the loss against the trainable parameters
        with tf.GradientTape() as tape:
            perdicted_actions =  self.b_actor(states)
            predicted_values = self.b_critic([states, perdicted_actions], training=True)
            actor_loss = -tf.math.reduce_mean(predicted_values)
            actor_grad = tape.gradient(actor_loss, self.b_actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.b_actor.trainable_variables))
        
        
    # update the weights of the target networks
    def update_weights(self):
        for a, b in range(zip(self.t_actor, self.b_actor)):
            a.assign(self.tau * b + (1 - self.tau) * a)
        
        for a, b in range(zip(self.t_critic, self.b_actor)):
            a.assign(self.tau * b + (1 - self.tau) * a)
            
    def _create_actor(self):
        # input layer is based on shape of state space
        inputs = tf.keras.layers.Input(shape=(self.num_states,))
        # create initial output layer
        out = tf.keras.layers.Dense(units=16, activation="relu")(inputs)
        # create layers for the number of layers specified in the constructor
        for _ in range(self.num_actor_layers):
            out =  tf.keras.layers.Dense(units=16, activation="relu")(out)
            
        # add final output layer to be the shape of the expected action
        out = tf.keras.layers.Dense(units=self.num_actions, activation="relu")(out)
        model = keras.Model(inputs=inputs, outputs=out)
        # return the created model
        return model
        
    
    def _create_critic(self):
        # perform preprocessing on state input
        state_inputs = tf.keras.layers.Input(shape=(self.num_states, ))
        state_out = tf.keras.layers.Dense(units=16, activation="relu")(state_inputs)
        state_out = tf.keras.layers.Dense(units=32, activation="relu")(state_out)
        
        # perform preprocessing on action input
        action_inputs = tf.keras.layers.Input(shape=(self.num_actions, ))
        action_out = tf.keras.layers.Dense(units=32, activation="relu")(action_inputs)
        # concatenate the state output and action outuput together
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        out = tf.keras.layers.Dense(16, activation="relu")(concat)
        for _ in range(self.num_critic_layers):
            out = tf.keras.layers.Dense(units=16, activation="relu")(out)
        
        out = tf.keras.layers.Dense(units=1, activation="tanh")(out)
        # create the model with the state and action inputs and the final value output
        model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=out)
        return model