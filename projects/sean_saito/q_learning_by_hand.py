import numpy as np
import time

def Q_learning_demmo():
    alpha = 1.0
    gamma = 0.8
    
    epsilon = 0.1
    num_of_episodes = 100
    R = np.array([
                            [-1, 0, -1, -1, -1, -1],
                            [ 0, -1, 0, -1, 0, -1],
                            [-1, 0, -1, -50, -1, -1],
                            [-1, -1, 0, -1, -1, -1],
                            [-1, 0, -1, -1, -1, 100],
                            [-1, -1, -1, -1, -1, -1]
                            ])
    
    # initialize the q table
    Q = np.random.uniform(size=(6, 6))
    
    # run for each episode
    for episode in range(num_of_episodes):
        # print("Episodes: ", episode)
        s = np.random.randint(0, 5)
        # print("Initial state: ", s)
        while s != 5:
            actions = [a for a in range(6) if R[s][a] != -1]
            # print("actions: ", actions)
            # epsilon greedy
            if np.random.random() <= epsilon:
                # print("Picked a random action")
                a = np.random.choice(actions)
            else:
                a = actions[np.argmax(Q[s][actions])]
            next_state = a
            # update q table
            Q[s][a] += alpha * (R[s][a] + gamma * np.max(Q[next_state]) - Q[s][a])
            
            # go to next state
            s = next_state
            # print("new state", s)
            
    return Q

start_time = time.time()
print(Q_learning_demmo())
total_time = time.time() - start_time

print("Total time: ", total_time)