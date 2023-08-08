from itertools import islice

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

g_counter = 0
base_velocity = 9.842

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
        self.last_distance_y_from_middle = 0
        self.last_score = 0
        self.client_socket = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=200, shape=(1, 4))

    def step(self, action):
        self.client_socket.send(struct.pack('i', action))
        data = self.client_socket.recv(1024)
        values = struct.unpack('ffffff', data)
        velocity = abs(values[0])
        bird_y_coordinate = values[1]
        gap_y_coordinate = values[2]
        score = values[3]
        gap_length = values[4]
        is_done = values[5]
        reward = 0.0

        if abs(bird_y_coordinate - gap_y_coordinate) > gap_length / 6.0:
            if bird_y_coordinate > gap_y_coordinate:
                if not action:
                    reward -= ((10*velocity) / base_velocity) * 40.0
                else:
                    reward += 10
            if bird_y_coordinate < gap_y_coordinate:
                if action:
                    reward -= ((10*velocity) / base_velocity) * 400.0 #bigger penalty for jumping when falling - becuase jumping resets the velocity downwords, which ruins big gap placement difference
                else:
                    reward += 10
        else:
            reward += 100.0


        # if abs(bird_y_coordinate - gap_y_coordinate) > self.last_distance_y_from_middle:
        #     reward = -10
        # else:
        #     reward += 10

        # if bird_y_coordinate < (gap_y_coordinate-(gap_length / 4.0)) and action:
        #     reward = -10
        # else:
        #     reward += 10
        #
        # if bird_y_coordinate > (gap_y_coordinate+(gap_length / 4.0)) and not action:
        #     reward = -10
        # else:
        #     reward += 10

        # if is_done:
        #     reward = -1000

        self.last_distance_y_from_middle = abs(bird_y_coordinate - gap_y_coordinate)
        self.last_score = score
        return (bird_y_coordinate, gap_y_coordinate, gap_length, velocity), reward, is_done, {}

    def reset(self):
        global g_counter
        self.last_distance_y_from_middle = 0
        self.last_score = 0
        g_counter += 1
        if g_counter == 5:
            subprocess.Popen(["./Game_GUI"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=os.getcwd())
            g_counter = 0
        else:
            subprocess.Popen(["./Game"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=os.getcwd())
        self.client_socket, _ = server_socket.accept()
        return 380.0, 360.0, 390.0, base_velocity


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
    checkpoint_path = './checkpoints/checkpoint'
    if os.path.exists(checkpoint_path):
        model.load_weights(tf.train.latest_checkpoint('./checkpoints'))
        print("using saved model")
    else:
        print("creating new model")
    model.summary()

    dqn = build_agent(model, action_space)

    fit = True
    if fit:
        dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
        dqn.save_weights(r'checkpoints/final.h5f', overwrite=True)
    server_socket.close()
