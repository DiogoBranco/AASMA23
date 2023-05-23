import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from game_objects import Item

class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def append(self, error, experience):
        self.buffer.append(experience)
        self.priorities.append(abs(error) + 1e-5)  # Avoid zero priority

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        probabilities = np.array(self.priorities) / sum(self.priorities)
        batch_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in batch_indices]
        return batch

    def update(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_shape, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, replay_buffer_size=2000):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(replay_buffer_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.model_input_shape = self.model.input_shape[1:]  # New line

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))  # Flatten layer to convert 2D state into 1D
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        next_state = np.reshape(next_state, (-1,) + self.model_input_shape)
        target = reward
        if not done:
            target += self.gamma * np.amax(self.model.predict(next_state)[0])
        td_error = abs(target - self.model.predict(state)[0][action])
        self.memory.append(td_error, (state, action, target, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        states, targets_f = [], []
        for state, action, target, next_state, done in minibatch:
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history

class Agent:
    def __init__(self, x, y, env, field_of_view, state_shape, action_size):
        self.x = x
        self.y = y
        self.env = env
        self.field_of_view = field_of_view
        self.dqn_agent = DQNAgent(state_shape, action_size)

    def act(self, state):
        return self.dqn_agent.act(state)

    def get_agent_view(self):
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

        return np.reshape(view_grid_transformed, (1,) + self.dqn_agent.model_input_shape)  # New line

    def replay(self, batch_size):
        self.dqn_agent.replay(batch_size)

    def step(self, action):
        raise NotImplementedError("This method should be overridden by child classes")

class Cop(Agent):
    def __init__(self, x, y, env, field_of_view, state_shape, action_size):
        super().__init__(x, y, env, field_of_view, state_shape, action_size)

    def step(self, action):
        # Get current state
        state = self.get_agent_view()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)

        # Try to move the agent and get the reward
        reward = self.env.move_agent(self, self.x + dx, self.y + dy)

        # Get the next state
        next_state = self.get_agent_view()

        done = self.env.is_game_over()
        self.dqn_agent.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}

class Thief(Agent):
    def __init__(self, x, y, env, field_of_view, state_shape, action_size):
        super().__init__(x, y, env, field_of_view, state_shape, action_size)

    def step(self, action):
        # Get current state
        state = self.get_agent_view()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)

        # Try to move the agent and get the reward
        reward = self.env.move_agent(self, self.x + dx, self.y + dy)

        # Get the next state
        next_state = self.get_agent_view()

        done = self.env.is_game_over()
        self.dqn_agent.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}


