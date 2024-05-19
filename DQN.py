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
    def __init__(self, input_channels, hidden_dim, output_dim):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate the size of the output from the convolutional layers
        dummy_input = torch.zeros(1, input_channels, 96, 96)
        conv_output_size = self.conv_stack(dummy_input).view(1, -1).size(1)
        
        self.fc_stack = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        x = self.conv_stack(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        outputs = self.fc_stack(x)
        return outputs

def sample_minibatch(replay_buffer, minibatch_size):
    if (len(replay_buffer) < minibatch_size):
        return None
    indices = np.random.choice(len(replay_buffer), minibatch_size, replace=False)
    minibatch = [replay_buffer[i] for i in indices]
    return minibatch

def eps_sceduler(episode, range_episode):
    min_eps = 0.1
    base = range_episode / 10
    multiplyer = 1.0 - 0.1
    epsilon = 1.0 - multiplyer * (episode / base)

    if epsilon < min_eps:
        epsilon = min_eps
    return epsilon

def train(network, env, video_folder, action_space):
    replay_buffer = []
    num_episodes = 1000
    T = 500
    gamma = 0.90
    optimizer = optim.Adam(network.parameters(), lr=0.05)
    
    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print('episode: ', episode)
        seed = np.random.randint(0, 10000)
        current_state, info = env.reset(options={"randomize": False})
        current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

        for t in range(T):
            if t % 250 == 0 and t != 0:
                print('t: ', t)
            sample = random.uniform(0, 1)
            epsilon = eps_sceduler(episode, num_episodes)
            if sample < epsilon:
                action_index = random.randint(0, len(action_space) - 1)
            else:
                q_values = network(current_state)
                action_index = torch.argmax(q_values).item()

            action = action_space[action_index]
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            done = terminated or truncated

            replay_buffer.append((current_state, action_index, reward, next_state, done))

            size_minibatch = 32
            minibatch = sample_minibatch(replay_buffer, size_minibatch)
            if minibatch is not None:
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.cat(states, dim=0)
                next_states = torch.cat(next_states, dim=0)
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
        
        # Step the scheduler at the end of each episode
        scheduler.step()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message="WARN: Overwriting existing videos at")

    env = gym.make('CarRacing-v2', render_mode='human')
    input_channels = env.observation_space.shape[2]
    hidden_dim = 512
    
    # Define a set of discrete actions for the continuous action space
    action_space = [
        np.array([0.0, 0.0, 0.0]),  # No action
        np.array([0.0, 1.0, 0.0]),  # Accelerate
        np.array([0.0, 0.0, 1.0]),  # Brake
        np.array([-1.0, 0.0, 0.0]), # Steer left
        np.array([1.0, 0.0, 0.0])   # Steer right
    ]
    output_dim = len(action_space)
    
    network = Qfunction(input_channels, hidden_dim, output_dim)

    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    train_folder = './videos/train'
    os.makedirs(train_folder, exist_ok=True)

    test_folder = './videos/test'
    os.makedirs(test_folder, exist_ok=True)

    train(network, env, train_folder, action_space)

    print('now testing')

    env = gym.make('CarRacing-v2', render_mode='rgb_array')

    test_episodes = 10
    for episode in range(test_episodes):
        video_file = f'run_{episode}'

        video_env = RecordVideo(env, video_folder=video_folder, name_prefix=video_file)
        
        seed = np.random.randint(0, 10000)
        state, info = video_env.reset(options={"randomize": False})
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            video_env.render()
            q_values = network(state)
            action_index = torch.argmax(q_values).item()
            action = action_space[action_index]
            state, reward, terminated, truncated, info = video_env.step(action)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if terminated or truncated:
                done = True

        video_env.close()

    env.close()