from agent import Cop, Thief
from game_objects import Item, Obstacle
import numpy as np


class Environment:
    def __init__(self, size, num_cops, num_thieves, num_items, num_obstacles):
        if size*size < num_cops + num_thieves + num_items + num_obstacles:
            raise ValueError("Not enough cells for the required objects")

        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        
        self.cops = [Cop(*self._place_on_grid(Cop)) for _ in range(num_cops)]
        self.thieves = [Thief(*self._place_on_grid(Thief)) for _ in range(num_thieves)]
        self.items = [Item(*self._place_on_grid(Item)) for _ in range(num_items)]
        self.obstacles = [Obstacle(*self._place_on_grid(Obstacle)) for _ in range(num_obstacles)]

    def _place_on_grid(self, entity_type):
        while True:
            x, y = np.random.randint(0, self.size, 2)
            if self.grid[x, y] is None:
                self.grid[x, y] = entity_type(x, y)
                return x, y
                
    def move(self, agent, new_x, new_y):
        if new_x >= self.size or new_y >= self.size or new_x < 0 or new_y < 0 or isinstance(self.grid[new_x, new_y], Obstacle):
            return False
        if isinstance(agent, Cop) and isinstance(self.grid[new_x, new_y], Thief):
            thief_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if thief_to_remove in self.thieves:
                self.thieves.remove(thief_to_remove)
        elif isinstance(agent, Thief) and isinstance(self.grid[new_x, new_y], Item):
            item_to_remove = self.grid[new_x, new_y]
            self.grid[new_x, new_y] = None
            if item_to_remove in self.items:
                self.items.remove(item_to_remove)
        self.grid[agent.x, agent.y] = None
        self.grid[new_x, new_y] = agent
        agent.x, agent.y = new_x, new_y
        return True

    def is_game_over(self):
        return len(self.thieves) == 0 or len(self.items) == 0
    
    def display(self):
        print('\n' + '-' * self.size * 4)
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