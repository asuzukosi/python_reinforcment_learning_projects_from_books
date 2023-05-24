import tensorflow as tf
import numpy as np

from dqn_impl.ReplayMemory import ReplayMemory

# implementation of the Actor, the actor is responsible for returnning
# the action that should be carried out by the agent


# convert this to be a sequential model in tensorflow 2
class ActorNetwork:
    def __init__(self, input_state, output_dim, hidden_layers, activation=tf.nn.relu):
        self.x = input_state
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        with tf.compat.v1.variable_scope('actor_network'):
            self.output = self._build()
            self.vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                    tf.compat.v1.get_variable_scope().name)
            
    
    def _build(self):
        layer = self.x
        init_b = tf.constant_initializer(0.01)
        
        for i, num_unit in enumerate(self.hidden_layers):
            layer = tf.compat.v1.layers.dense(layer, num_unit, 
                                              init_b=init_b, 
                                              name=f"hidden_layer_{i}")
            
            output = tf.compat.v1.layers.dense(layer, self.output_dim, activation=self.activation, init_b=init_b, name="output")
            
        return output
    

# convert this to a sequential neural network that concatinates the input and the action and accepts that as input
# it also has only one linear output
# the critic network takes both the state and the action as input
# it performs some transformation on the input before appending it to the action linearly
# the final output is the q value of the state action pair
class CriticNetwork:
    
    def __init__(self, input_state, input_action, hidden_layers):
        assert len(hidden_layers) > 2
        self.input_state = input_state
        self.input_action = input_action
        self.hidden_layers = hidden_layers
        
        with tf.compat.v1.variable_scope('critic_network'):
            self.output = self._build()
            self.vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 
                                                    tf.compat.v1.get_variable_scope().name)
    
    def _build(self):
        layer = self.input_state
        init_b = tf.constant_initializer(0.01)
        
        for i, num_unit in enumerate(self.hidden_layers):
            if i != 1:
                layer = tf.compat.v1.layers.dense(layer, num_unit, init_b=init_b, name=f"hidden_layer_{i}")
            else:
                layer = tf.concat([layer, self.input_action],
                                  axis=1, name="concat_action")
                layer = tf.compat.v1.layers.dense(layer, num_unit, init_b=init_b, name=f"hidden_layer_{i}")
            
        output = tf.compat.v1.layers.dense(layer, 1, activation=None, init_b=init_b, name="output")
        return tf.reshape(output, shape=(-1))


# compbines the actor and the critic network into one architecture
class ActorCriticNetwork:
    def __init__(self, input_dim, action_dim, 
                 critic_layers, acotr_layers, 
                 actor_activation, scope="ac_network"):
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.scope = scope
        
        self.x = tf.compat.v1.placeholder(shape=(None, input_dim), dtype=tf.float32, name="x")
        self.y = tf.compat.v1.placeholder(shape=(None, ), dtype=tf.float32, name="y")
        
        with tf.compat.v1.variable_scope(scope):
            self.actor_network = ActorNetwork(self.x, action_dim, hidden_layers=acotr_layers, activation=actor_activation)
            
            self.critic_network = CriticNetwork(self.x, self.actor_network.get_output_layer(), hidden_layers=critic_layers)
            self.vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys, tf.compat.v1.get_variable_scope().name)
            self._build()
            
    def _build(self):
        value =  self.critic_network.get_output_layer()
        actor_loss = -tf.reduce_mean(value)
        self.actor_vars = self.actor_network.get_params()
        self.actor_grad = tf.gradients(actor_loss, self.actor_vars)
        tf.summary.scalar('actor_loss', self.actor_loss, collections=["actor"])
        self.actor_summary = tf.compat.v1.summary.merge_all("actor")
        
        critic_loss = 0.5 * tf.reduce_mean(tf.square((value - self.y)))
        self.critic_vars = self.critic_network.get_params()
        self.critic_grad = tf.gradients(critic_loss, self.critic_vars)
        tf.summary.scalar('critic_loss', critic_loss, collections=["critic"])
        self.critic_summary = tf.compat.v1.summary.merge_all("critic")
        
    def get_action(self, sess, state):
        return self.actor_network.get_action(sess, state)
    
    def get_value(self, sess, state):
        return self.critic_network.get_value(sess, state)
    
    def get_action_value(self, sess, state, action):
        return self.critic_network.get_action_value(sess, state, action)
    
    def get_actor_feed_dict(self, state):
        return {self.x: state}
    
    def get_critic_feed_dict(self, state, action, target):
        return {self.x: state, self.y:target, 
                self.critic_network.input_action: action}
        
    def get_clone_op(self, network, tau=0.9):
        update_ops = []
        new_vars = {v.name.replace(network.scope, '') for v in network.vars}
        
        for v in self.vars:
            u = (1 - tau) * v + tau * new_vars[v.name.replace(self.scope, '')]
            update_ops.append(tf.assign(v, u))
        return update_ops
    
    
class DPG:
    def __init__(self, config, task, directory, callback=None, summary_writer=None):
        self.task = task
        self.directory = directory
        self.callback = callback
        self.summary_writer = summary_writer
        
        self.config = config
        self.batch_size = config["batch_size"]
        self.n_episodes = config["n_episodes"]
        self.capacity = config["capacity"]
        self.history_len = config["history_len"]
        
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.time_between_two_copies = config["time_between_two_copies"]
        self.update_interval = config["update_interval"]
        self.tau = config["tau"]
        
        self.action_dim = task.get_action_dim()
        self.state_dim = task.get_state_dim() * self.history_len
        self.critic_layers = [50, 50]
        self.actor_layers = [50, 50]
        
        self.actor_activation = task.get_activation_fn()
        
        self._init_modules()
        
        
    def _init_modules(self):
        # initiate replay memory
        self.replay_memory = ReplayMemory(capacity=self.capacity, 
                                          history_len=self.history_len)
        
        # initiate behavioral actor critic network
        self.ac_network = ActorCriticNetwork(input_dim=self.state_dim, action_dim=self.action_dim, 
                                             critic_layers=self.critic_layers, acotr_layers=self.actor_layers, 
                                             actor_activation=self.actor_activation, scope="ac_network")
        
        # initiate the target network
        self.target_network = ActorCriticNetwork(input_dim=self.state_dim, action_dim=self.action_dim, 
                                                 critic_layers=self.critic_layers, actor_layers=self.actor_layers,
                                                actor_activation=self.actor_activation, scope="target_network")
        
        # operation for updating the target network
        self.clone_op = self.target_network.get_clone_op(self.ac_network, tau=self.tau)
        
        # initialize tensorboard information store
        self.t_score = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="new_score")
        
        tf.summary.scalar('score', self.t_score, collections=['dpg'])
        self.summary_op = tf.compat.v1.summary.merge_all('dpg')
    
    # lets the agent know which action to pic
    def choose_action(self, sess, state, epsilon=0.1):
        x = np.asarray(np.expand_dims(state, axis=0), dtype=np.float32)
        action = self.ac_network.get_action(sess, x)[0]
        return action + epsilon * np.random.randn(len(action))
    
    # simulates a state transition
    def play(self, action):
        r, new_state, termination = self.task.play_action(action)
        return r, new_state, termination
    
    # this method is used to update the target networks
    def update_target_network(self, sess):
        sess.run(self.clone_op)
        
        
    # train the 4 neural networks
    def train(self, sess, saver=None):
        num_of_trials = -1
        for episode in range(self.n_episodes):
            (frame, _) = self.task.reset()
            self.replay_memory.add_null_operations(frame, 0, 0, 0)
        
            for _ in range(self.config["T"]):
                num_of_trials += 1
                epsilon = self.epsilon_min + max(self.epsilon_decay - num_of_trials, 0)/self.epsilon_decay * (1 - self.epsilon_min)
                
                if num_of_trials % self.update_interval == 0:
                    self.optimizer.train_one_step(sess, num_of_trials, self.batch_size)
                
                state = self.replay_memory.get_full_state_from_observation(frame)
                action = self.choose_action(sess, state, epsilon)
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                if num_of_trials % self.time_between_two_copies == 0:
                    self.update_target_network(sess)
                    self.save(sess, saver)
                if self.callback:
                    self.callback()
                    
                if termination:
                    score = self.task.get_total_reward()
                    summary_str = sess.run(self.summary_op, feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str, num_of_trials)
                    self.summary_writer.flush()
                    break
                
    def evaluate(self, sess):
        for episode in range(self.n_episodes):
            frame = self.task.reset()
            self.replay_memory.add_null_operations(frame)
            
            for _ in range(self.config["T"]):
                print(f"Episode {episode}, total reward: {self.task.get_total_reward()}")

                state = self.replay_memory.get_full_state_from_observation(frame)
                
                action = self.choose_action(sess, state, self.epsilon_min)
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                if self.callback:
                    self.callback()
                
                if termination:
                    break
        
    
