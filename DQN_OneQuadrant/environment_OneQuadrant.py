import math
import numpy as np


class OneQuadrant:
    def __init__(self):
        self.inductor = 100.0
        self.resistor = 50.0
        self.voltageIn = 10.0
        self.voltageOut = 0.0
        self.timeStep = 0.04
        self.i = 0.0
        self.i_dot = 0.0
        self.referenceVal = 5.0

        #Actions Space
        self.action_space = np.array([0,1])

        #Observation Space
        self.observation_space = np.array([0, self.voltageIn])

        self.state = 0.0
        self.episode_ended = False
        self.steps_episode = 0

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def reset(self):
        self.i = np.random.uniform(low = 0, high = 0.2)
        self.state = self.i * self.resistor
        self.episode_ended = False
        self.i_dot = 0.0
        self.steps_episode = 0

        return self.state

    def step(self, action):
        # Verifying that the action is valid
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert action == 0 or action == 1, err_msg

        if self.episode_ended:
            return self.reset()
        
        if action == 1:
            self.i_dot = self.voltageIn/self.inductor - (self.resistor/self.inductor)*self.i
            #using integration euler procedure
            self.i +=  self.timeStep * self.i_dot
            self.voltageOut = self.i * self.resistor
            self.state = self.voltageOut
        elif action == 0:
            self.i_dot = -(self.resistor/self.inductor)*self.i
            self.i += self.timeStep * self.i_dot
            self.voltageOut = self.i * self.resistor
            self.state = self.voltageOut
        else:
            raise ValueError('action should be 0 or 1')

        # Reward definition
        # R = 1 - abs(w - y)/maxVal
        reward = 1 - abs(self.referenceVal - self.voltageOut)/self.voltageIn
        # count steps in the episode
        self.steps_episode += 1

        #Evaluation for episode ending
        if self.steps_episode > 100:
            self.episode_ended = True
        
        #Return the trajectory
        return self.state, reward, self.episode_ended, {}


#Example of initialization
'''
env = OneQuadrant()
print(env.action_space)
state = env.reset()
print(state)
action_close = 1

while not env.episode_ended:
    state, reward, done, _ = env.step(action_close)
    print(state, reward, done)
'''