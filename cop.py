from agent import Agent
from thief import Thief
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np

class Cop(Agent):
    def __init__(self, id, fov, speed, model, env):
        super().__init__(id, fov, speed, model, env)

        # Assuming you are storing all states in a list
        self.coop_state_size = (2*self.fov + 1) * (2*self.fov + 1) * (1 + len(self.env.cops))

    def act_coop(self, state):
        # Flatten the state representation into a 1D array
        flattened_state = state.flatten().reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(flattened_state)
        return np.argmax(act_values[0])  # returns action

    def _build_coop_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.coop_state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def replay_coop(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.flatten().reshape(1, -1)  # Flattening state array
            next_state = next_state.flatten().reshape(1, -1)  # Flattening next_state array
            target = self.model.predict(state)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def random_move(self, excludes):
        return super().random_move_aux(excludes)
    
    def greedy_move(self, excludes):
        return super().greedy_move_aux(excludes, Thief)

    def learning_move(self, excludes):
        return super().learning_move_aux(excludes)
