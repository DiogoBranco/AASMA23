from agent import Cop, Thief
from game_objects import Item, Obstacle
from gym import spaces
import numpy as np
import gym

class Environment(gym.Env):
    def __init__(self, size, num_cops, num_thieves, num_items, num_obstacles):
        super(Environment, self).__init__()

        if size*size < num_cops + num_thieves + num_items + num_obstacles:
            raise ValueError("Not enough cells for the required objects")

        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        
        self.action_space = spaces.Discrete(4)
        state_shape = ((2*3+1), (2*3+1))
        
        self.cops = [self._place_on_grid(Cop, 3, state_shape, self.action_space.n) for _ in range(num_cops)]
        self.thieves = [self._place_on_grid(Thief, 3, state_shape, self.action_space.n) for _ in range(num_thieves)]
        self.items = [self._place_on_grid(Item) for _ in range(num_items)]
        self.obstacles = [self._place_on_grid(Obstacle) for _ in range(num_obstacles)]

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size, 1), dtype=np.int)

    def _place_on_grid(self, entity_type, field_of_view=3, state_shape=None, action_size=None):
        unoccupied_cells = np.argwhere(self.grid == None)
        if unoccupied_cells.size == 0:
            raise ValueError("The grid is full, cannot place more entities.")
        idx = np.random.choice(unoccupied_cells.shape[0])
        x, y = unoccupied_cells[idx]
        if entity_type in [Cop, Thief]:
            entity = entity_type(x, y, self, field_of_view, state_shape, action_size)
        else:
            entity = entity_type(x, y)
        self.grid[x, y] = entity
        return entity

    def step(self):
        actions = [agent.act(agent.get_agent_view()) for agent in (self.cops + self.thieves)]
        results = [agent.step(action) for action, agent in zip(actions, self.cops + self.thieves)]
        new_states, rewards, dones, _ = zip(*results)

        if self.is_game_over():
            dones = [True]*len(dones)
            
        return new_states, rewards, dones, {}

    def reset(self):
        self.grid = np.empty((self.size, self.size), dtype=object)
        self._reset_positions(self.cops)
        self._reset_positions(self.thieves)
        self._reset_positions(self.items)
        self._reset_positions(self.obstacles)
        return [agent.get_agent_view() for agent in (self.cops + self.thieves)]

    def _reset_positions(self, entities):
        for entity in entities:
            unoccupied_cells = np.argwhere(self.grid == None)
            idx = np.random.choice(unoccupied_cells.shape[0])
            x, y = unoccupied_cells[idx]
            self.grid[entity.x, entity.y] = None  # Clear old position
            self.grid[x, y] = entity
            entity.x = x
            entity.y = y

    def _action_to_direction(self, action):
        if action == 0:
            return -1, 0  # up
        elif action == 1:
            return 0, 1  # right
        elif action == 2:
            return 1, 0  # down
        elif action == 3:
            return 0, -1  # left
        else:
            raise ValueError(f"Invalid action {action}")

    def move_agent(self, agent, new_x, new_y):
        if isinstance(self.grid[new_x, new_y], Obstacle):
            return -1
        # Handle catching a thief by a cop
        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Thief):
            self.grid[new_x, new_y] = None
            return 1

        # Handle stealing an item by a thief
        elif isinstance(agent, Thief) and isinstance(self.grid[new_x, new_y], Item):
            self.grid[new_x, new_y] = None
            return 1

        # Moving the agent
        self.grid[agent.x, agent.y] = None
        self.grid[new_x, new_y] = agent
        agent.x, agent.y = new_x, new_y
        return 0

    def is_game_over(self):
        return len(self.thieves) == 0 or len(self.items) == 0

    def render(self, mode='human'):
        # Render the environment to the screen
        if mode == 'human':
            for i in range(self.size):
                for j in range(self.size):
                    cell = self.grid[i, j]
                    if isinstance(cell, Cop):
                        print('| C ', end='')
                    elif isinstance(cell, Thief):
                        print('| T ', end='')
                    elif isinstance(cell, Item):
                        print('| I ', end='')
                    elif isinstance(cell, Obstacle):
                        print('| O ', end='')
                    else:
                        print('|   ', end='')
                print('|')
                print('-' * self.size * 4)
        else:
            super(Environment, self).render(mode=mode)  # Just in case super has render for 'rgb_array' or 'ansi' modes

    def close(self):
        # Perform any necessary cleanup
        pass
