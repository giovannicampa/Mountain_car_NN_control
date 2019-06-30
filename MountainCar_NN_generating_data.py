import gym
import random
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


env = gym.make("MountainCar-v0")

nr_episodes = 20000
time_steps = 200


# To store the successful training data and the corresponding scores
training_data = np.array([])
training_data.shape = (0,3)
accepted_scores = []

# playing all the episodes
for episode in range(nr_episodes):
    env.reset()
    print('Episode nr: {}' .format(episode))
    score = 0
    game_memory = []
    previous_observation = []

    # playing through one episode's time steps with random actions
    for t in range(time_steps):
        # env.render()
        action = random.randrange(0, 3)
        observation, reward, done, info = env.step(action)

        if len(previous_observation) > 0:
           game_memory.append(previous_observation.tolist() + [action])

        # we take observation[0] > -0.2 as an sign that we are doing well and give a positive reward for it
        if observation[0] >= -0.2:
            reward = 1


        # the current observation becomes the previous observation of the next time step
        previous_observation = observation
        score = score + reward

        if done:
            break

    # once finished the episode, we check if the score is satisfying. If so we save the data
    if score >= -198:
        accepted_scores.append(score)
        training_data = np.concatenate((training_data, game_memory[len(game_memory)-180:len(game_memory)]))


# Splitting the data in train and test set
X = training_data[:, [0,1]]
y = training_data[:, [2]]

np.savetxt('X.csv', X, delimiter = ",")
np.savetxt('y.csv', y, delimiter = ",")




