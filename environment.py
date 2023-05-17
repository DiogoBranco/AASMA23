from agent import Cop, Thief
from game_objects import Item, Obstacle
from utils import nearest_item, nearest_thief
import numpy as np
import gym
from gym import spaces



class Environment(gym.Env):
    def __init__(self, size, num_cops, num_thieves, num_items, num_obstacles, num_states, num_actions):
        super(Environment, self).__init__()

        if size*size < num_cops + num_thieves + num_items + num_obstacles:
            raise ValueError("Not enough cells for the required objects")

        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        
        self.cops = [self._place_on_grid(Cop, num_states, num_actions) for _ in range(num_cops)]
        self.thieves = [self._place_on_grid(Thief, num_states, num_actions) for _ in range(num_thieves)]
        self.items = [self._place_on_grid(Item) for _ in range(num_items)]
        self.obstacles = [self._place_on_grid(Obstacle) for _ in range(num_obstacles)]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous
        self.action_space = spaces.Discrete(4)

        # Observation is the grid itself
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.size, self.size, 1), dtype=np.int)

    def _place_on_grid(self, entity_type, num_states=None, num_actions=None):
        while True:
            x, y = np.random.randint(0, self.size, 2)
            if self.grid[x, y] is None:
                if entity_type in [Cop, Thief]:
                    entity = entity_type(x, y, self, num_states, num_actions)
                else:
                    entity = entity_type(x, y)
                self.grid[x, y] = entity
                return entity

                
    def step(self, agent):
        # Get current state
        state = self._get_state(agent)

        # Agent chooses action
        action = agent.choose_action(state)
        dx, dy = self._action_to_direction(action)

        # Execute action and get reward
        new_x = agent.x + dx
        new_y = agent.y + dy
        reward = self.move(agent, new_x, new_y)

        # If the action was invalid (reward is negative), choose a new action
        while reward < 0:
            action = agent.choose_action(state)
            dx, dy = self._action_to_direction(action)
            new_x = agent.x + dx
            new_y = agent.y + dy
            reward = self.move(agent, new_x, new_y)

        # Get next state
        next_state = self._get_state(agent)

        # Agent updates Q-values
        agent.update(state, action, reward, next_state)

        # Check if the episode has ended
        done = self.is_game_over()

        return reward, done

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
        # Reset the state of the environment to an initial state
        self.grid = np.empty((self.size, self.size), dtype=object)
        self.cops = [Cop(*self._place_on_grid(Cop)) for _ in range(len(self.cops))]
        self.thieves = [Thief(*self._place_on_grid(Thief)) for _ in range(len(self.thieves))]
        self.items = [Item(*self._place_on_grid(Item)) for _ in range(len(self.items))]
        self.obstacles = [Obstacle(*self._place_on_grid(Obstacle)) for _ in range(len(self.obstacles))]

        return self._get_observation()

    def _action_to_direction(self, action):
        # Convert action index to movement
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        return directions[action]

    def _calculate_reward(self):
        # Calculate the reward based on the current state of the environment
        # Modify this method according to your reward scheme
        reward = 0
        if len(self.thieves) == 0:
            reward += 100  # High reward for catching all thieves
        if len(self.items) == 0:
            reward += 50   # Moderate reward for collecting all items
        return reward

    def move(self, agent, new_x, new_y):
        if new_x >= self.size or new_y >= self.size or new_x < 0 or new_y < 0 or isinstance(self.grid[new_x, new_y], Obstacle):
            return -1  # Penalty for trying to move out of the grid or into an obstacle

        # Prevent a cop from moving to a cell that is already occupied by another cop
        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Cop):
            return -1  # Penalty for trying to move into a cell occupied by another cop

        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Thief):
            thief_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if thief_to_remove in self.thieves:
                self.thieves.remove(thief_to_remove)
            return 1  # Reward for catching a thief

        elif isinstance(agent, Thief) and isinstance(self.grid[new_x, new_y], Item):
            item_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if item_to_remove in self.items:
                self.items.remove(item_to_remove)
            return 1  # Reward for stealing an item

        self.grid[agent.x, agent.y] = None
        self.grid[new_x, new_y] = agent
        agent.x, agent.y = new_x, new_y
        return 0  # No reward or penalty for moving to an empty cell


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

    