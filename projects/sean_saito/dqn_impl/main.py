from DQN import Agent
from ReplayMemory import ReplayMemory, convert_to_grayscale, crop_image_to_84_84, normalize_image
from QNetwork import QNetworkBuilder
import gymnasium as gym

# create behavioural and target networks
behavioral_network = QNetworkBuilder(input_shape=(84, 84, 4), n_outputs=4, learning_rate=0.00075).build()
target_network = QNetworkBuilder(input_shape=(84, 84, 4), n_outputs=4, learning_rate=0.00075).build()

# create replay memory and specify the transformations
memory = ReplayMemory(transformations=[convert_to_grayscale, normalize_image, crop_image_to_84_84])
atari = gym.make("Breakout-v4")
atari_eval = gym.make("Breakout-v4", render_mode="human")

agent = Agent(target_model=target_network, behaviour_model=behavioral_network, repaly_memory=memory, environment=atari)
agent.train()
# agent.evaluate(atari_eval)