import gym
import random
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam


env = gym.make("MountainCar-v0")

# Loading the training data and splitting it in training and testing set
X = np.genfromtxt('X.csv', delimiter = ',')
y = np.genfromtxt('y.csv', delimiter = ',')

# 1. INSTANTIATE
enc = OneHotEncoder()

# 2. FIT
enc.fit(y.reshape(-1,1))

# 3. Transform
y_onehot = enc.transform(y.reshape(-1,1)).toarray()


# ======= Keras approach =======
# Initialise the model
input_size = len(X[0])
output_size = len(y_onehot[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=input_size, activation='relu'))
model.add(tf.keras.layers.Dense(52, activation='relu'))
model.add(tf.keras.layers.Dense(output_size, activation='linear'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mse',
              metrics = ['accuracy'])

# Train the model
model.fit(X, y_onehot, epochs = 10)


# Here we will play the game with the actions calculated by the regressor
duration = 10000
nr_games = 10
for j in range(nr_games):
    env.reset()
    for time_steps in range(duration):
        env.render()
        # First action is random, then predicted by the NN
        if time_steps == 0:
            action = random.randrange(0,3)
            observation, reward, done, info = env.step(action)
        else:
            action = model.predict(previous_observation.reshape(1, -1))
            observation, reward, done, info = env.step(np.argmax(action))
        previous_observation = observation
        print(time_steps)

        if done:
            break
