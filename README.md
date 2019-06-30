# Mountain_car_NN_control
Solving the Mountain Cart Problem with a supervised learning approach.

Files contained: 

- MountainCar_random_action:
    having the car do random actions to understand how the environment works

- MountainCar_NN_generating_data:
    The reward function has been modified in a way that a reward is given also when the end goal is approached and not only reached.
    The data (X: observations), (y: action) is selected from the runs with a higher score.

- MountainCar_NN_training:
    A Neural Network is trained with the X and y data (which is now 1hot encoded)
    At the end some games are played in which the action is the one predicted by the network basing on the previous observation
