# import gymnasium
import gymnasium as gym

# create gymnasium environment
env = gym.make('Reacher-v4', render_mode="human")

env.reset()

for i in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()

