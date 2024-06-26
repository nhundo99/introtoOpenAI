from torch import nn
import random
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime
import warnings


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
    if len(replay_buffer) < minibatch_size:
        return None
    indices = np.random.choice(len(replay_buffer), minibatch_size, replace=False)
    minibatch = [replay_buffer[i] for i in indices]
    return minibatch

def train(network, env, video_folder):
    replay_buffer = []
    num_episodes = 30
    T = 500
    epsilon = 0.3
    gamma = 0.99
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    for episode in range(num_episodes):
        seed = np.random.randint(0, 10000)
        current_state, info = env.reset(seed=seed)
        current_state = torch.tensor(current_state, dtype=torch.float32)

        for t in range(T):
            sample = random.uniform(0, 1)
            if sample < epsilon:
                action = env.action_space.sample()
            else:
                q_values = network(current_state)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = terminated or truncated

            replay_buffer.append((current_state, action, reward, next_state, done))

            size_minibatch = 5
            minibatch = sample_minibatch(replay_buffer, size_minibatch)
            if minibatch is not None:
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.stack(states, dim=0)
                next_states = torch.stack(next_states, dim=0)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.uint8)
                actions = torch.tensor(actions, dtype=torch.long)

                current_q_values = network(states)
                next_q_values = network(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                yi = rewards + gamma * (1 - dones) * max_next_q_values

                loss_fn = torch.nn.MSELoss()
                action_values = current_q_values.gather(1, actions.unsqueeze(1))
                loss = loss_fn(action_values, yi.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_state = next_state
            if done:
                break

if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="WARN: Overwriting existing videos at")

    env = gym.make('CartPole-v1', render_mode='human')
    input_dim = env.observation_space.shape[0]
    hidden_dim = 64
    output_dim = env.action_space.n
    network = Qfunction(input_dim, hidden_dim, output_dim)

    # Create a video folder
    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    # train folder
    train_folder = './videos/train'
    os.makedirs(train_folder, exist_ok=True)

    # train folder
    test_folder = './videos/test'
    os.makedirs(test_folder, exist_ok=True)


    train(network, env)

    print('now testing')

    # Wrap the environment with RecordVideo
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    test_episodes = 10
    for episode in range(test_episodes):
        video_file = f'run_{episode}'

        # Wrap the environment with RecordVideo
        video_env = RecordVideo(env, video_folder=video_folder, name_prefix=video_file)
        
        state, info = video_env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
            video_env.render()
            q_values = network(state.unsqueeze(0))
            action = torch.argmax(q_values).item()
            state, reward, terminated, truncated, info = video_env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            if terminated or truncated:
                done = True

        video_env.close()

    env.close()