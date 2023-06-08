from agent import Agent
from item import Item
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np

class Thief(Agent):
    def __init__(self, id, fov, speed, model, num_thieves, env):
        super().__init__(id, fov, speed, model, env)

        # Assuming you are storing all states in a list
        self.coop_state_size = (2*self.fov + 1) * (2*self.fov + 1) * num_thieves
        self.model_coop = self._build_coop_model()
        self.target_model_coop = self._build_coop_model()
        self.update_target_coop_model()

    def act_coop(self, state):
        # Flatten the state representation into a 1D array
        flattened_state = state.flatten().reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_coop.predict(flattened_state)
        return np.argmax(act_values[0])  # returns action

    def _build_coop_model(self):
        # Neural Net for Deep-Q learning Model
        model_coop = Sequential()
        model_coop.add(Dense(24, input_dim=self.coop_state_size, activation='relu'))
        model_coop.add(Dense(24, activation='relu'))
        model_coop.add(Dense(self.action_size, activation='linear'))
        model_coop.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model_coop

    def replay_coop(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.flatten().reshape(1, -1)  # Flattening state array
            next_state = next_state.flatten().reshape(1, -1)  # Flattening next_state array
            target = self.model_coop.predict(state)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model_coop.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model_coop.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_coop_model(self):
        # copy weights from model to target_model
        self.target_model_coop.set_weights(self.model_coop.get_weights())

    def learning_move_coop(self, excludes):
        # Get the cooperative state of the environment grid (including other agents of the same type)
        state = self.env.get_state(self)
        state = np.array(state).reshape(1, -1)
        # Use the act_coop method to choose an action
        action_idx = self.act_coop(state)
        # Map the output of the act_coop method to a valid action
        actions = ["up", "down", "left", "right", "stay"]
        chosen_action = actions[action_idx]
        return (chosen_action, excludes)

    def random_move(self, excludes):
        return super().random_move_aux(excludes)
    
    def greedy_move(self, excludes):
        return super().greedy_move_aux(excludes, Item)

    def learning_move(self, excludes):
        return super().learning_move_aux(excludes)
