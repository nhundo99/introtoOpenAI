from torch import nn
import random
import numpy as np
import gymnasium as gym


class Qfunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        outputs = self.stack(x)
        return outputs
    


def train(network, env):
    # initialize replay buffer
    replay_buffer = []

    num_episodes = 10
    T = 10
    epsilon = 0.2
    for episode in range(num_episodes):
        #initialise sequence
        current_state = env.reset()

        for t in range(T):
            # select an action
            sample = random.uniform(0,1)
            if sample < epsilon:
                # ranodm action
                action = env.action_space.sample()
            else:
                # action that maximizes the Q function
                q_values = network(current_state)
                action = np.argmax(q_values)

            # execute the action
            next_state, reward, terminated, truncated, info = env.step(action)

            # update replay buffer
            replay_buffer.append(current_state, action, reward, next_state)

            # sample minibatch from buffer
            



# network for Q function
# input the state and output value for each possible action

# maybe consider normalization or feature scaling

# main function
# initialize empty replay buffer with size N
# for loop episode
# initialize sequence
# # for loop t
# select action
# execute action
# new sequence
# store transition in D
# sample a minibatch from replay buffer D
# calculate yi
# do gradient descent step