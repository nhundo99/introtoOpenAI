import gymnasium as gym

# create environment
env = gym.make('CartPole-v1', render_mode='human')

# initialize the environment
observation, info = env.reset(seed=42)

for _ in range(1000):
    # render the environment
    env.render()
    
    # select a random action
    action = env.action_space.sample()

    # take the action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        # reset if done
        observation, info = env.reset()

env.close()


"""

Want to implemetn a simple Q learning algorithm and maybe also policy iteration to understand basics

"""