from torch import nn
import random
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim


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
    

def sample_minibatch(replay_buffer, minibatch_size):
    # check if buffer has enough elements
    if len(replay_buffer) < minibatch_size:
        return None
    
    # random sample of indicies
    indicies = np.random.choice(len(replay_buffer), minibatch_size, replace=False)

    minibatch = [replay_buffer[i] for i in indicies]

    return minibatch
    


def train(network, env):
    # initialize replay buffer
    replay_buffer = []

    num_episodes = 10
    T = 10
    epsilon = 0.2
    gamma = 0.99

    optimizer = optim.Adam(network.parameters(), lr=0.001)


    for episode in range(num_episodes):
        #initialise sequence
        current_state, info = env.reset()
        # print("Initial state:", current_state)
        current_state = torch.tensor(current_state, dtype=torch.float32)

        for t in range(T):
            # select an action
            sample = random.uniform(0,1)
            if sample < epsilon:
                # ranodm action
                action = env.action_space.sample()
            else:
                # action that maximizes the Q function
                q_values = network(current_state)
                action = torch.argmax(q_values).item()

            # execute the action
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = terminated or truncated 

            # update replay buffer
            replay_buffer.append((current_state, action, reward, next_state, done))

            # sample minibatch from buffer
            size_minibatch = 5
            minibatch = sample_minibatch(replay_buffer, size_minibatch)
            if minibatch is not None:
                states, actions, rewards, next_states, dones = zip(*minibatch)
                # print("States before conversion:", states)
                states = torch.stack(states, dim=0)
                next_states = torch.stack(next_states, dim=0)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.uint8)
                actions = torch.tensor(actions, dtype=torch.long)

                # compute yi
                current_q_values = network(states)
                next_q_values = network(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                yi = rewards + gamma * (1 - dones) * max_next_q_values

                # compute loss
                loss_fn = torch.nn.MSELoss()
                action_values = current_q_values.gather(1, actions.unsqueeze(1))
                loss = loss_fn(action_values, yi.unsqueeze(1))

                # gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_state = next_state

            

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')

    # Define network dimensions based on the environment
    input_dim = env.observation_space.shape[0]
    hidden_dim = 64
    output_dim = env.action_space.n

    # Initialize the Q-function network
    network = Qfunction(input_dim, hidden_dim, output_dim)

    # Train the network
    train(network, env)

    # Test the trained network
    test_episodes = 10
    for _ in range(test_episodes):
        state = np.array(env.reset(), dtype=np.float32)
        print("Initial state:", state)
        state = torch.tensor(state)
        done = False
        while not done:
            env.render()
            q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(q_values).item()
            state, _, done, _ = env.step(action)
        env.close()