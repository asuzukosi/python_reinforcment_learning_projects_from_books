import tensorflow as tf
import numpy as np
from collections import deque
import random

class QNetwork:
    def __init__(self, input_shape=(84, 84, 4), n_outputs=4, network_type="cnn", scope="q_network"):
        self.width, self.height, self.channels = input_shape
        self.n_outputs = n_outputs
        self.network_type = network_type
        self.scope = scope
        
        # imput image provided to the network
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.channels, self.width, self.height))
        # estimated q value provided by the network
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
        # selected action
        self.a = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
        
        with tf.variable_scope(scope):
            self.build()
            self.build_loss()
            
    def build(self):
        self.net = {} # initiate network dictionary
        self.net["input"] = tf.transpose(self.x, (0, 2, 3, 1)) # performs transformation permutations along the axis provided in the input tuple
        init_b = tf.constant_initializer(0.01)
        
        if self.network_type == "cnn":
            self.net["conv1"] = tf.compat.v1.layers.conv2d(self.net["input"], 32, kernel=(8,8), stride=(4, 4), init_b=init_b, name="conv1")
            self.net["conv2"] = tf.compat.v1.layers.conv2d(self.net["conv1"], 64, kernel=(4,4), stride=(2, 2), init_b=init_b, name="conv2")
            self.net["conv3"] = tf.compat.v1.layers.conv2d(self.net["conv2"], 64, kernel=(3,3), stride=(1, 1), init_b=init_b, name="conv3")
            self.net["feature"] = tf.compat.v1.layers.dense(self.net["conv3"], 512, init_b=init_b, name="fc1")
            
        elif self.network_type == "cnn_nips":
            self.net["conv1"] = tf.compat.v1.layers.conv2d(self.net["input"], 16, kernel=(8,8), stride=(4, 4), init_b=init_b, name="conv1")
            self.net["conv2"] = tf.compat.v1.layers.conv2d(self.net["conv1"], 32, kernel=(8,8), stride=(4, 4), init_b=init_b, name="conv2")
            self.net["feature"] = tf.compat.v1.layers.dense(self.net["conv2"], 512, init_b=init_b, name="fc1")
            
            # add remaining layers
        elif self.network_type == "mlp":
            # add multi layer perceptron layers
            self.net["fc1"] = tf.compat.v1.layers.dense(self.net["input"], 50, init_b=init_b, name="fc1")
            self.net["feature"] = tf.compat.v1.layers.dense(self.net["fc1"], 512, init_b=init_b, name="fc2")
            
        else:
            raise NotImplementedError("Unknown network type")
        
        self.net["values"] = tf.compat.v1.layers.dense(self.net["feature"], self.n_output, activation=None, init_b=init_b, name="values")
        self.net["q_values"] = tf.reduce_max(self.net["values"], axis=1, name="q_values")
        self.net["q_action"] = tf.argmax(self.net["q_values"], axis=1, name="q_action", output_type=tf.int32)
        self.vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, tf.compat.v1.get_variable_scope().name)
        
        
    def build_loss(self):
        # get indices
        indices = tf.compat.v1.placeholder(tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=0))
        # get values
        value = tf.gather_nd(self.net["values"], indices)
        self.loss = 0.5 * tf.reduce_mean(tf.square(value - self.y))
        # get gradients of trainable parameters
        self.gradient = tf.gradients(self.loss, self.vars)
        # add summary for tensorboard
        tf.compat.v1.summary.scalar("loss", self.loss, collections=["q_network"])
        # merge all summaries for q_network together
        self.summary_op = tf.compat.v1.summary.merge_all("q_network")
        
        
    def get_q_value(self, sess, state):
        return sess.run(self.net['q_value'], feed_dict={self.x: state})
    
    def get_q_action(self, sess, state):
        return sess.run(self.net['q_action'], feed_dict={self.x: state})
    
    def get_feed_dict(self, states, actions, values):
        return {self.x: states, self.a: actions, self.y: values}
                
    def get_clone_op(self, network):
        new_vars = {v.name.replace(network.scope, ''): v for v in network.vars}
        return [tf.compat.v1.assign(v, new_vars[v.name.replace(self.scope, '')]) for v in self.vars]


class ReplayMemory:
    def __init__(self, history_len=4, capacity=1_000_000, batch_size=32, input_scale=255.0):
        self.capacity = capacity
        self.history_length = history_len
        self.batch_size = batch_size
        self.input_scale = input_scale
        
        # set frames and others and deques
        self.frames = deque([])
        self.others = deque([])
    
    # adds new experience to the replay memory
    def add(self, frame, action, r, termination):
        if len(self.frames) >= self.capacity:
            self.frames.popleft()
            self.others.popleft()
        self.frames.append(frame)
        self.others.append((action, r, termination))
    
    # adds a no action state to the memory
    def add_null_ops(self, init_frame):
        for _ in range(self.history_length):
            self.add(init_frame, 0, 0, 0, 0)
            
    def phi(self, new_frame):
        assert len(self.frames) >= self.history_length # assert that you have at least 4 frames
        images = [new_frame] + [self.frames[-1-i] for i in range(len(self.history_length - 1))] # get previous images 
                                                                                                # and add them to form state
        return np.concatenate(images, axis=0)
    
    def sample(self):
        while True:
            index = random.randint(a=(self.history_length-1), b=len(self.frames)-2)
            infos = [self.others[index-i] for i in range(self.history_length)]
            
            # check if terminated before index
            flag = False
            for i in range(1, self.history_length):
                if infos[i][2] == 1: # if a state is terminal within the sequence
                    flag = True
                    break
                
            if flag:
                continue
            
            # construct state and new state and eperience actions, reward and termination
            state = self._phi(index)
            new_state = self._phi(index+1)
            action, reward, termination = self.others[index]
            state = np.asarray(state / self.input_scale, dtype=np.float32)
            new_state = np.asarray(new_state / self.input_scale, dtype=np.float32)
                
            return (state, action, reward, new_state, termination)
    
    # stacks images from index
    def _phi(self, index):
        images = [self.frames[index + i] for i in range(len(self.history_length))]
        return np.concatenate(images, axis=0)
    

class Optimizer:
    def __init__(self, config, feedback_size, q_network: QNetwork, target_network: QNetwork, replay_memory:ReplayMemory):
        self.feedback_size = feedback_size
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        
        self.gama = config["gama"]
        self.num_frames = config["num_frames"]
    
        optimizer = create_optimizer(config["optimizer"], config["learning_rate"], config["rho"], config["rmsprop_epsilon"])
        self.train_op = optimizer.apply_gradients(zip(self.q_network.gradient, self.q_network.vars))


    def sample_transitions(self, sess, batch_size):
        w, h = self.feedback_size
        states = np.zeros((batch_size, self.num_frames, w, h), dtype=np.float32)
        new_states = np.zeros((batch_size, self.num_frames, w, h), dtype=np.float32)
        targets = np.zeros(batch_size, dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.float32)
        terminations = np.zeros(batch_size, dtype=np.float32)
        
        
        for i in range(batch_size):
            state, action, reward, new_state, termination = self.replay_memory.sample()
            states[i] = state
            new_states[i] = new_state
            targets[i] = reward
            actions[i] = action
            terminations[i] = termination
            
            targets[i] += self.gamma * (1 - terminations[i]) * self.target_network.get_q_value(sess, new_state)
            
        # at the end of all that calculation return states actions and tergets
        return states, actions, targets
    
    
    def train_one_step(self, sess, step, batch_size):
        states, actions, targets = self.sample_transitions(sess, batch_size)
        feed_dict = self.q_network.get_feed_dict(states, actions, targets)
        if self.summary_writer and step % 100 == 0:
            summary_str, _ = sess.run([self.summary_writer, self.train_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, step)
            self.summary_writer.flush()
            
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            
            
# We now combine these modules together to have our full DQN implementation

class DQN:
    def __init__(self, config, game, directory, callback=None, summary_writer=None):
        self.game = game
        self.actions = game.get_available_actions()
        self.feedback_size = game.get_feedback_size()
        self.callback = callback
        self.summary_writer = summary_writer
        
        self.config = config # config contains all the important information of the dqn
        self.batch_size = config["batch_size"]
        self.n_episodes = config["num_episodes"]
        self.capacity = config["capacity"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.num_frames = config["num_frames"]
        self.num_nullops = config["num_nullops"]
        self.time_between_two_copies = config["time_between_two_copies"]
        self.input_scale = config["input_scale"]
        self.update_interval = config["update_interval"]
        self.directory = directory
        
        self._init_modules()
        
        
    def train(self, sess, saver=None):
        num_of_trials = -1
        for episode in range(self.n_episodes):
            self.game.reset()
            frame = self.game.get_current_frame()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
                
            for _ in range(self.config["T"]):
                num_of_trials += 1
                epsilon_greedy = self.epsilon_min + max(self.epsilon_decay  - num_of_trials, 0) / self.epsilon_decay * (1 - self.epsilon_min)
                
                if num_of_trials % self.update_interval == 0:
                    self.optimizer.train_one_step(sess, num_of_trials, self.batch_size)
                
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, epsilon_greedy)
                new_frame, reward, terminated, truncated, info = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                if num_of_trials % self.time_between_two_copies == 0:
                    self.update_target_network(sess)
                    self.save(sess, saver)
                
                if self.callback:
                    self.callback()
                if termination:
                    score = self.game.get_total_reward()
                    summary_str = sess.run(self.summary_op, feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str, num_of_trials)
                    self.summary_writer.flush()
                    break
                
    def evaluate(self, sess):
        for episode in range(self.n_episodes):
            self.game.reset()
            frame = self.game.get_current_feedback()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
            
            for _ in range(self.config["T"]):
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, self.epsilon_min)
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                
                if self.callback:
                    self.callback()
                    if termination:
                        break
                    
        
    
            
        
        
        
    
        
        
        
            
    
        
        
        
        
        

            
            