from agent import Cop, Thief
from game_objects import Item, Obstacle
from utils import nearest_item, nearest_thief
import numpy as np
import gym
from gym import spaces



class Environment(gym.Env):
    def __init__(self, size, num_cops, num_thieves, num_items, num_obstacles):
        super(Environment, self).__init__()

        if size*size < num_cops + num_thieves + num_items + num_obstacles:
            raise ValueError("Not enough cells for the required objects")

        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        
        self.cops = [self._place_on_grid(Cop, 3) for _ in range(num_cops)] # 3 as an example field of view
        self.thieves = [self._place_on_grid(Thief, 3) for _ in range(num_thieves)] # 3 as an example field of view
        self.items = [self._place_on_grid(Item) for _ in range(num_items)]
        self.obstacles = [self._place_on_grid(Obstacle) for _ in range(num_obstacles)]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous
        self.action_space = spaces.Discrete(4)

        # Observation is the grid itself
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.size, self.size, 1), dtype=np.int)

    def _place_on_grid(self, entity_type, field_of_view=3):
        unoccupied_cells = np.argwhere(self.grid == None)
        if unoccupied_cells.size == 0:
            raise ValueError("The grid is full, cannot place more entities.")
        idx = np.random.choice(unoccupied_cells.shape[0])
        x, y = unoccupied_cells[idx]
        if entity_type in [Cop, Thief]:
            entity = entity_type(x, y, self, field_of_view)
        else:
            entity = entity_type(x, y)
        self.grid[x, y] = entity
        return entity

                
    def step(self):
        done = False
        rewards = []
        new_states = []
        dones = []
        
        for cop in self.cops:
            action = cop.choose_action(cop.get_state())
            new_state, reward, done, _ = cop.step(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)

        for thief in self.thieves:
            action = thief.choose_action(thief.get_state())
            new_state, reward, done, _ = thief.step(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)

        if self.is_game_over():
            done = True

        return new_states, rewards, dones, {}

    def _get_state(self, agent):
        # Define the state as the relative position of the agent and its target.
        if isinstance(agent, Cop):
            target = nearest_thief(self, agent)
        elif isinstance(agent, Thief):
            target = nearest_item(self, agent)
        else:
            return None

        if target is None:
            # If there is no target, return a default state.
            return 0

        # Calculate the relative positions.
        dx = agent.x - target.x
        dy = agent.y - target.y

        # Convert the relative positions to a single index.
        # This assumes that the maximum size of the grid is 100.
        # You should adjust this according to your actual grid size.
        state = dx * 100 + dy
        return state
    

    def reset(self):
        self.grid = np.empty((self.size, self.size), dtype=object)
        self._reset_positions(self.cops)
        self._reset_positions(self.thieves)
        self._reset_positions(self.items)
        self._reset_positions(self.obstacles)
        return [agent.get_state() for agent in (self.cops + self.thieves)]

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
        # Convert action index to movement
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        return directions[action]


    def calculate_reward(self, agent, new_x, new_y):
        if new_x >= self.size or new_y >= self.size or new_x < 0 or new_y < 0 or isinstance(self.grid[new_x, new_y], Obstacle):
            return -1  # Penalty for trying to move out of the grid or into an obstacle

        # Prevent a cop from moving to a cell that is already occupied by another cop
        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Cop):
            return -1  # Penalty for trying to move into a cell occupied by another cop

        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Thief):
            return 1  # Reward for catching a thief

        elif isinstance(agent, Thief) and isinstance(self.grid[new_x, new_y], Item):
            return 1  # Reward for stealing an item

        return 0  # No reward or penalty for moving to an empty cell

    def move_agent(self, agent, new_x, new_y):
        # Handle catching a thief by a cop
        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Thief):
            thief_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if thief_to_remove in self.thieves:
                self.thieves.remove(thief_to_remove)

        # Handle stealing an item by a thief
        elif isinstance(agent, Thief) and isinstance(self.grid[new_x, new_y], Item):
            item_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if item_to_remove in self.items:
                self.items.remove(item_to_remove)

        # Moving the agent
        self.grid[agent.x, agent.y] = None
        self.grid[new_x, new_y] = agent
        agent.x, agent.y = new_x, new_y

    def move(self, agent, new_x, new_y):
        reward = self.calculate_reward(agent, new_x, new_y)
        if reward >= 0:  # The agent can move
            self.move_agent(agent, new_x, new_y)
        return reward

    def _get_observation(self):
        # Convert the current grid into an observation suitable for the observation space
        # This is a placeholder implementation, you may need to modify it to suit your needs
        observation = np.zeros((self.size, self.size, 1), dtype=np.int)
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                if isinstance(cell, Cop):
                    observation[i, j, 0] = 1
                elif isinstance(cell, Thief):
                    observation[i, j, 0] = 2
                elif isinstance(cell, Item):
                    observation[i, j, 0] = 3
                elif isinstance(cell, Obstacle):
                    observation[i, j, 0] = 4
        return observation


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

    