import numpy as np
import tensorflow as tf
import gym
import random
from tensorflow import keras

import socket
import os
import subprocess
import struct
from tensorflow.keras.callbacks import EarlyStopping
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Convolution2D, Activation, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from gym.spaces import Discrete, Box


def build_agent(model, nb_actions):
    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.2)
    nb_steps_warmup = 1000
    target_model_update = .2
    gamma = .99
    lr = .0001
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                   target_model_update=target_model_update, policy=policy, gamma=gamma)
    dqn.compile(Adam(learning_rate=lr), metrics=['mae'])

    return dqn


def build_model(nb_states, nb_actions):
    model = Sequential()
    model.add(Dense(128, input_shape=nb_states))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    model.add(Flatten())

    return model


# Define the Flappy Bird game environment using OpenAI Gym
class FlappyBirdEnvironment(gym.Env):
    def __init__(self):
        self.last_y = 0
        self.client_socket = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=200, shape=(1, 5))

    def step(self, action):
        self.client_socket.send(struct.pack('i', action))
        data = self.client_socket.recv(1024)
        values = struct.unpack('ffffff', data)
        reward = 0
        print(values)
        print()

        if values[3] < 720/2 and values[3]-self.last_y >0.0:
            print ("!!!!!!!!!!!!!!!!!!!!!!!!")
            reward += 100
        if values[3] > 720 / 2 and values[3] - self.last_y < 0.0:
            print ("!!!!!!!!!!!!!!!!!!!!!!!!")
            reward += 100
        if abs(values[1] - values[2]) < 25:
            reward += 100
        elif abs(values[1] - values[2]) < 45:
            reward += 50
        elif abs(values[1] - values[2]) < 65:
            reward += 20
        # elif values[5] == 1.0:
        #     reward = -100
        else:
            reward = -50

        if abs(values[3]) < 100:
            reward -= 100
        else:
            reward += 100
        if abs(values[4]) < 100:
            reward -= 100
        else:
            reward += 100

        self.last_y = values[3]
        return (values[0], values[1], values[2], values[3], values[4]), reward, values[5], {}

    def reset(self):
        subprocess.Popen(["./Game"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.getcwd())
        self.client_socket, _ = server_socket.accept()
        return -9.841968536376953, 1021.2803955078125, 1147.7086181640625, 0.0, 720.0


    # def choose_action(self, state):
    #     return 0 if random.uniform(0, 1) > 0.05 else 1


# Main function to run the Flappy Bird game and train the agent
if __name__ == "__main__":

    # create socket
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    socket_file = '/tmp/my_socket'
    if os.path.exists(socket_file):
        os.remove(socket_file)
    server_socket.bind(socket_file)
    server_socket.listen(1)

    # Create the Flappy Bird environment
    env = FlappyBirdEnvironment()

    # Define the state shape and action space size
    state_shape = env.observation_space.shape  # Three input features: distance upper, distance lower, velocity, distance between pillars
    action_space = env.action_space.n # Two actions: 0 (do not click on the jump button) and 1 (click on the jump button)

    # Create the agent
    model = build_model(state_shape, action_space)
    model.summary()

    dqn = build_agent(model, action_space)

    fit = True
    if fit:
        dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
        date_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        dqn.save_weights(r'checkpoints/%s/dqn_weights.h5f' % date_time, overwrite=True)
    server_socket.close()
