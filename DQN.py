# network for Q function

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