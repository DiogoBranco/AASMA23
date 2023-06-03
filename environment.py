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
        self.grid = np.full((size, size), None, dtype=object) # Change from np.empty to np.full to initialize with None
        
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
        y, x = unoccupied_cells[idx]  # Change to ensure correct indexing
        if entity_type in [Cop, Thief]:
            entity = entity_type(x, y, self, field_of_view, state_shape, action_size)
        else:
            entity = entity_type(x, y)
        if self.grid[y, x] == None:
            self.grid[y, x] = entity  # Change to ensure correct indexing
        return entity

    def step(self):
        actions = [agent.act(agent.get_agent_view()) for agent in (self.cops + self.thieves)]
        results = [agent.step(action) for action, agent in zip(actions, self.cops + self.thieves)]
        new_states, rewards, dones, _ = zip(*results)

        if self.is_game_over():
            dones = [True]*len(dones)
            
        return new_states, rewards, dones, {}

    def reset(self):
        padding_size = max(agent.field_of_view for agent in (self.cops + self.thieves))
        self.grid = np.full((self.size + 2 * padding_size, self.size + 2 * padding_size), None, dtype=object)
        self._reset_positions(self.cops, padding_size)
        self._reset_positions(self.thieves, padding_size)
        self._reset_positions(self.items, padding_size)
        self._reset_positions(self.obstacles, padding_size)
        return [agent.get_agent_view() for agent in (self.cops + self.thieves)]

    def _reset_positions(self, entities, padding_size):
        for entity in entities:
            unoccupied_cells = np.argwhere(self.grid == None)
            if unoccupied_cells.size == 0:
                raise ValueError("The grid is full, cannot reset positions.")
            idx = np.random.choice(unoccupied_cells.shape[0])
            y, x = unoccupied_cells[idx]
            if self.grid[y, x] == None:
                self.grid[entity.y, entity.x] = None
                self.grid[y, x] = entity
                entity.y, entity.x = y + padding_size, x + padding_size

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

    def move_agent(self, agent, new_y, new_x):  # Change to ensure correct indexing
        # Boundary check
        if new_y < 0 or new_y >= self.grid.shape[0] or new_x < 0 or new_x >= self.grid.shape[1] or isinstance(self.grid[new_y, new_x], Obstacle):
            return -1  # Invalid move

        # Handle catching a thief by a cop
        if isinstance(agent, Cop) and isinstance(self.grid[new_y, new_x], Thief):
            self.grid[new_y, new_x] = agent
            self.grid[agent.y, agent.x] = None
            agent.y, agent.x = new_y, new_x
            return 1

        # Handle stealing an item by a thief
        elif isinstance(agent, Thief) and isinstance(self.grid[new_y, new_x], Item):
            self.grid[new_y, new_x] = agent
            self.grid[agent.y, agent.x] = None
            agent.y, agent.x = new_y, new_x
            return 1

        # Moving the agent
        if self.grid[new_y, new_x] == None:
            self.grid[agent.y, agent.x] = None
            self.grid[new_y, new_x] = agent
            agent.y, agent.x = new_y, new_x
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

