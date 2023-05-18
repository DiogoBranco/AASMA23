import numpy as np
from gym import spaces

class Agent:
    def __init__(self, x, y, env, field_of_view):
        self.x = x
        self.y = y
        self.env = env
        self.field_of_view = field_of_view
        self.state_space = spaces.Box(low=0, high=4, shape=(2*self.field_of_view+1, 2*self.field_of_view+1, 1), dtype=np.int)
        self.action_space = spaces.Discrete(4)
        self.q_table = np.zeros((self.state_space.shape[0] * self.state_space.shape[1], self.action_space.n))
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

    def get_state(self):
        # Here, we return the part of the environment grid that represents
        # the agent's current field of view as the state
        view_grid = self.env.grid[max(0, self.y - self.field_of_view):min(self.env.height, self.y + self.field_of_view + 1),
                                  max(0, self.x - self.field_of_view):min(self.env.width, self.x + self.field_of_view + 1)]
        return view_grid.flatten()

    def get_location(self):
        return self.x, self.y

    def choose_action(self):
        state = self.get_state()
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = self.action_space.sample()  # Explore action space
        else:
            action = np.argmax(self.q_table[state])  # Exploit learned values
        return action

    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
                                      self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state]))

class Cop(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)

    def step(self, action):
        # Get current state
        state = self.get_state()
        
        if action == 0:   # Move right
            self.x += 1
        elif action == 1: # Move left
            self.x -= 1
        elif action == 2: # Move down
            self.y += 1
        elif action == 3: # Move up
            self.y -= 1

        # Ensure the cop doesn't move out of the grid.
        self.x = max(0, min(self.x, self.env.width - 1))
        self.y = max(0, min(self.y, self.env.height - 1))

        # Get new state and reward
        new_state = self.get_state()
        reward = self.env.get_reward(self.x, self.y)

        # Update Q-table for Q(s,a)
        self.update_q_table(state, action, reward, new_state)
        
        return new_state, reward

class Thief(Agent):
    def __init__(self, x, y, env, field_of_view):
        super().__init__(x, y, env, field_of_view)
    
    def step(self, action):
        # Get current state
        state = self.get_state()

        if action == 0:  # move up
            self.y -= 1
        elif action == 1:  # move down
            self.y += 1
        elif action == 2:  # move left
            self.x -= 1
        elif action == 3:  # move right
            self.x += 1

        # Ensure the thief doesn't move out of the grid.
        self.x = max(0, min(self.x, self.env.width - 1))
        self.y = max(0, min(self.y, self.env.height - 1))

        # Get new state and reward
        new_state = self.get_state()
        reward = self.env.get_reward(self.x, self.y)

        # Update Q-table for Q(s,a)
        self.update_q_table(state, action, reward, new_state)

        return new_state, reward

