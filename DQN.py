from torch import nn

class Qfunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16,2)
        )


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