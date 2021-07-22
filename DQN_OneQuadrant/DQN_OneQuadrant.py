#from DQN_OneQuadrant.environment_OneQuadrant import OneQuadrant
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

from environment_OneQuadrant import OneQuadrant


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    #X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation=None, kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='OneQuadrant_DQN_model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.01, rho=0.90, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model

class DQNAgent:
    """
    Description:
        The following environment represents a LR circuit with 
        a control switch.
        The system always starts with the Inductor Not charge
        The goal is to keep the charge at the output in a specific
        value, by open or closing the switch, i.e., by controlling
        the charge in the inductor

    Observation:
        Type: float32
        Num   Observation             Min   Max
        0     Voltage at the output   0     10

    Actions:
        Type: Discrete (2)
        Num     Action
        0       Open the Switch
        1       Close the Switch

    Reward:
        The reward will be calculated according the difference 
        between the reference value and the actual observation
        Maximun reward available is 1, worst reward availables are 0.5
        reward = 1 - abs(self.referenceVal - self.voltageOut)/self.voltageIn

    Starting State:
        Always start with Voltage 0

    Episode Termination:
        Episode Length is greater than 100

    """
    def __init__(self):
        self.env = OneQuadrant()
        # One Quadrant has a maximun steps of 100
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 400

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(x=(state,)))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = []
        next_state = []
        #state = np.zeros((self.batch_size, self.state_size))
        #next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state.append(minibatch[i][0])
            #state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state.append(minibatch[i][3])
            #next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(np.array(state), target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
            
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            #state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0       #represent the steps
            episode_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.reshape(next_state, [1, self.state_size])
                #if not done or i == self.env._max_episode_steps-1:
                #    reward = reward
                #else:
                #    reward = -100
                
                #Average Reward
                episode_reward += reward

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, step: {}, e: {:.2}, ep_reward: {}".format(e, self.EPISODES, i, self.epsilon, episode_reward))
                    if e > 25:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        return
                self.replay()

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            #state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            episode_reward = 0
            while not done:
                action = np.argmax(self.model.predict(x=(state,)))
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                #state = np.reshape(next_state, [1, self.state_size])
                i += 1
                print("observation: {}, action_taked: {}".format(next_state, action))
                if done:
                    print("episode: {}/{}, avg_reward: {}".format(e, self.EPISODES, episode_reward))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    #agent.test()
    
