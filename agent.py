import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from gym import spaces
from game_objects import Item

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
        self.memory.append((np.copy(state), action, reward, np.copy(next_state), done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([val[0] for val in minibatch], dtype=np.float32)
        actions = np.array([val[1] for val in minibatch], dtype=np.float32)
        rewards = np.array([val[2] for val in minibatch], dtype=np.float32)
        next_states = np.array([(np.zeros(self.state_space.shape)
                                if val[4] else val[3]) for val in minibatch], dtype=np.float32)

        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.train_on_batch(states, targets_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def step(self, action):
        raise NotImplementedError("This method should be overridden by child classes")

class Cop(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

    def step(self, action):

        print(f"Cop at ({self.x}, {self.y}) received action: {action}")
        # Get current state
        state = self.get_state()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)
        print(f"Converted action to direction: dx={dx}, dy={dy}")

        # Try to move the agent
        reward = self.env.move(self, self.x + dx, self.y + dy)
        print(f"Attempted to move cop to ({self.x + dx}, {self.y + dy}). Reward received: {reward}")

        # If the move was unsuccessful, try other actions
        if reward == -1:
            for new_action in range(self.action_space.n):
                if new_action != action:
                    dx, dy = self.env._action_to_direction(new_action)
                    reward = self.env.move(self, self.x + dx, self.y + dy)
                    if reward != -1:
                        break

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

    def choose_action(self, state):
        # Check for immediate objective (thief) in agent's adjacent cells
        for action in range(self.action_space.n):
            dx, dy = self.env._action_to_direction(action)
            new_x, new_y = self.x + dx, self.y + dy
            if self.env._is_within_grid(new_x, new_y) and isinstance(self.env.grid[new_y][new_x], Thief):
                print(f"Immediate objective (thief) at ({new_x}, {new_y}), choosing action: {action}")
                return action

        # If no immediate objective, use model to choose action or explore randomly
        if np.random.rand() <= self.epsilon:
            print("Choosing random action due to epsilon")
            return self.action_space.sample()
        act_values = self.model.predict(state)
        chosen_action = np.argmax(act_values[0])
        print(f"Chosen action: {chosen_action}, Q-values: {act_values[0]}")
        return chosen_action

class Thief(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

    def step(self, action):
        print(f"Thief at ({self.x}, {self.y}) received action: {action}")

        # Get current state
        state = self.get_state()

        # Convert action into a direction
        dx, dy = self.env._action_to_direction(action)
        print(f"Converted action to direction: dx={dx}, dy={dy}")

        # Try to move the agent
        reward = self.env.move(self, self.x + dx, self.y + dy)
        print(f"Attempted to move thief to ({self.x + dx}, {self.y + dy}). Reward received: {reward}")

        # If the move was unsuccessful, try other actions
        if reward == -1:
            for new_action in range(self.action_space.n):
                if new_action != action:
                    dx, dy = self.env._action_to_direction(new_action)
                    reward = self.env.move(self, self.x + dx, self.y + dy)
                    if reward != -1:
                        break

        # If an item is found, remove it
        items_to_remove = []
        for item in self.env.items[:]:  # Assuming the environment keeps a list of items
            if item.x == self.x + dx and item.y == self.y + dy:  
                items_to_remove.append(item)
                reward = 1  # Define your reward for stealing an item
                print(f"Found item at ({item.x}, {item.y}). Reward is now: {reward}")

        for item in items_to_remove:
            self.env.remove_item(item)

        # Get the next state
        next_state = self.get_state()

        done = self.env.is_game_over()
        self.remember(state, action, reward, next_state, done)

        return next_state, reward, done, {}

    def choose_action(self, state):
        # Check for immediate objective (item) in agent's field of view
        for action in range(self.action_space.n):
            dx, dy = self.env._action_to_direction(action)
            new_x, new_y = self.x + dx, self.y + dy
            if self.env._is_within_grid(new_x, new_y) and isinstance(self.env.grid[new_y][new_x], Item):
                # Check if moving to the item would place the thief adjacent to a cop
                for dx2, dy2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_x, adj_y = new_x + dx2, new_y + dy2
                    if self.env._is_within_grid(adj_x, adj_y) and isinstance(self.env.grid[adj_y][adj_x], Cop):
                        break  # Moving to the item would place the thief adjacent to a cop, so don't choose this action
                else:
                    print(f"Immediate objective (item) at ({new_x}, {new_y}), choosing action: {action}")
                    return action

        # If no immediate objective, use model to choose action or explore randomly
        if np.random.rand() <= self.epsilon:
            print("Choosing random action due to epsilon")
            return self.action_space.sample()
        act_values = self.model.predict(state)
        chosen_action = np.argmax(act_values[0])
        print(f"Chosen action: {chosen_action}, Q-values: {act_values[0]}")
        return chosen_action
