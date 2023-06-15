# https://github.com/sourcecode369/deep-reinforcement-learning/blob/master/DQN/CartPole-v1/CartPole-v1%20with%20Fixed%20Q%20Targets%20in%20TensorFlow.ipynb
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import math 
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.record_video import RecordVideo
import os

EPISODES = 50

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.003
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    def build_model(self):
        model = Sequential()
        model.add(Dense(32,input_dim=self.state_size,activation="relu",kernel_initializer="he_uniform"))
        model.add(Dense(32,activation="relu",kernel_initializer="he_uniform"))
        model.add(Dense(self.action_size,activation="linear",kernel_initializer="he_uniform"))
        model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model
    def remember(self,state,action,reward, next_state, done):
        self.memory.append((state,action,reward, next_state, done))
    def act(self,state):
        if np.random.rand()<self.epsilon:
            return np.random.choice(np.arange(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        replay_batch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in replay_batch:
            target_val = reward
            if not done:
                target_val = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target = self.model.predict(state)
            target[0][action] = target_val
            #print(target)
            self.model.fit(state, target, epochs = 1, verbose = 0)
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
            
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(env, './Output',  episode_trigger = lambda episode_number: False)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    batch_size = 5000
    scores = []
    
    after_training = os.path.dirname(os.path.realpath(__file__))+"/Output/after_training_dqn.mp4"
    video = VideoRecorder(env, after_training)
    
    for e in range(EPISODES):
        state, info = env.reset()
        state = np.reshape(state,[1,state_size])
        score = 0
        render_start = False
        render_end = False
        for time_p in range(500000):
            if render_start:
                env.render()
               
            action = agent.act(state)
            # next_state, reward, done, info = env.step(action)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = np.reshape(next_state,[1,state_size])
            reward = reward if not done else -10
            agent.remember(state,action,reward,next_state, done)
            score += reward
            state = next_state
            agent.replay(batch_size)
            video.capture_frame()
            if done:
                agent.update_target_model()
                scores.append(score)
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, score, agent.epsilon))
                break
        if render_end:            
            env.close()
video.close() 
#         if e % 10 == 0:
#              agent.save("./cartpole-dqn.h5")
               