import numpy as np
import random
from game_objects import Item
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from gym import spaces

class Agent:
    def __init__(self, x, y, env, field_of_view):
        self.x = x
        self.y = y
        self.env = env
        self.field_of_view = field_of_view
        self.state_space = spaces.Box(low=0, high=4, shape=(2*self.field_of_view+1, 2*self.field_of_view+1, 1), dtype=np.int)
        self.action_space = spaces.Discrete(4)
        self.model = self.build_model()
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=(self.state_space.shape[0]*self.state_space.shape[1],)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model


    def get_state(self):
        padded_grid = np.pad(self.env.grid, self.field_of_view, mode='constant', constant_values=0)
        view_grid = padded_grid[self.y:self.y + 2*self.field_of_view + 1,
                        self.x:self.x + 2*self.field_of_view + 1]

        # Transform grid items into numeric values
        view_grid_transformed = np.zeros_like(view_grid, dtype=np.float32)
        for i in range(view_grid.shape[0]):
            for j in range(view_grid.shape[1]):
                if isinstance(view_grid[i, j], Cop):
                    view_grid_transformed[i, j] = 1
                elif isinstance(view_grid[i, j], Thief):
                    view_grid_transformed[i, j] = 2
                elif isinstance(view_grid[i, j], Item):
                    view_grid_transformed[i, j] = 3
                else:
                    view_grid_transformed[i, j] = 0

        return np.reshape(view_grid_transformed, [1, -1])




    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.copy(state), action, reward, np.copy(next_state), done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state[None, :], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def step(self, action):
        raise NotImplementedError("This method should be overridden by child classes")

class Cop(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

    def step(self, action):
        # Get current state
        state = self.get_state()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)

        # Try to move the agent
        reward = self.env.move(self, self.x + dx, self.y + dy)

         # If a thief is caught, remove it
        thieves_to_remove = []
        for thief in self.env.thieves[:]:  # Assuming the environment keeps a list of thieves
            if thief.x == self.x + dx and thief.y == self.y + dy:  
                thieves_to_remove.append(thief)
                reward = 1  # Define your reward for catching a thief

        for thief in thieves_to_remove:
            self.env.remove_thief(thief)

        # Get the next state
        next_state = self.get_state()

        done = self.env.is_game_over()
        self.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}

class Thief(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

    def step(self, action):
        # Get current state
        state = self.get_state()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)

        # Try to move the agent
        reward = self.env.move(self, self.x + dx, self.y + dy)

        # If an item is found, remove it
        items_to_remove = []
        for item in self.env.items[:]:  # Assuming the environment keeps a list of items
            if item.x == self.x + dx and item.y == self.y + dy:  
                items_to_remove.append(item)
                reward = 1  # Define your reward for stealing an item

        for item in items_to_remove:
            self.env.remove_item(item)

        # Get the next state
        next_state = self.get_state()

        done = self.env.is_game_over()
        self.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}

