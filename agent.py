import numpy as np
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
        self.memory = []

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_space.shape)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def get_state(self):
        padded_grid = np.pad(self.env.grid, self.field_of_view, mode='constant', constant_values=0)
        view_grid = padded_grid[self.y:self.y + 2*self.field_of_view + 1,
                        self.x:self.x + 2*self.field_of_view + 1]
        return np.reshape(view_grid, [1, self.state_space.shape[0], self.state_space.shape[1], 1])


    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def step(self, action):
        # Get current state
        state = self.get_state()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)

        # Get the reward from the environment for the proposed move
        reward = self.env.move(self, self.x + dx, self.y + dy)
        
        if reward >= 0:
            # Only update position if the move was valid (reward >= 0)
            self.x += dx
            self.y += dy

        # Get the next state
        next_state = self.get_state()

        done = self.env.is_game_over()
        self.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}

class Cop(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

class Thief(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

