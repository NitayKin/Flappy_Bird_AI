import numpy as np
import tensorflow as tf
import gym
import random
import socket
import os
import subprocess
import struct


# Define the Flappy Bird game environment using OpenAI Gym
class FlappyBirdEnvironment(gym.Env):
    def __init__(self):
        self.client_socket = 0

    def step(self, action):
        reward = 5
        self.client_socket.send(struct.pack('i', action))
        data = self.client_socket.recv(1024)
        values = struct.unpack('ffff', data)
        return [values[0], values[1], values[2]], reward, values[3]

    def reset(self):
        subprocess.Popen(["./Game"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        self.client_socket, _ = server_socket.accept()


# Define your reinforcement learning agent
class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space

        # Define your DQN model using TensorFlow
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=self.state_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, state):
        return 0

    def train(self, state, action, next_state, reward, done):
        # Implement the agent's training process using Q-learning
        # You should use the DQN algorithm with experience replay
        pass


# Main training loop
def train_agent(env, agent, num_episodes=1000, batch_size=32, target_update_freq=100):
    for episode in range(num_episodes):
        state = env.reset()
        done = 0.0
        total_reward = 0

        while done == 0.0:
            # Choose action using the agent's policy
            action = agent.choose_action(state)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Update the agent's Q-function and train the DQN model
            agent.train(state, action, next_state, reward, done)

            # Update state and total reward for the next iteration
            state = next_state
            total_reward += reward

        # Print progress after each episode
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

        # Optionally update the target network periodically for more stable training
        if episode % target_update_freq == 0:
            # Update the target model (if you are using a DQN with a target network)
            pass

    # Save the trained model for later use, if needed
    agent.model.save("trained_model.h5")


# Main function to run the Flappy Bird game and train the agent
if __name__ == "__main__":

    #create socket
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    socket_file = '/tmp/my_socket'
    if os.path.exists(socket_file):
        os.remove(socket_file)
    server_socket.bind(socket_file)
    server_socket.listen(1)

    # Create the Flappy Bird environment
    env = FlappyBirdEnvironment()

    # Define the state shape and action space size
    state_shape = (4,)  # Four input features: distance upper, distance lower, velocity, distance between pillars
    action_space = 2    # Two actions: 0 (do not click on the jump button) and 1 (click on the jump button)

    # Create the agent
    agent = DQNAgent(state_shape=state_shape, action_space=action_space)

    # Train the agent
    train_agent(env, agent)


    server_socket.close()
