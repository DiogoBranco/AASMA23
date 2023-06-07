import random
import numpy as np
from thief import Thief
from cop import Cop
from item import Item
from obstacle import Obstacle

class Environment:
    def __init__(self, size, num_thieves, thieves_fov, thieves_speed, num_cops, cops_fov, cops_speed, num_items, num_obstacles, model):

        if size * size < num_thieves + num_cops + num_items + num_obstacles:
            raise ValueError("Too many entities for the grid size")

        self.size = size

        self.thieves = [Thief(id, thieves_fov, thieves_speed, model, self) for id in range(num_thieves)]
        self.cops = [Cop(id, cops_fov, cops_speed, model, self) for id in range(num_cops)]
        self.items = [Item(id) for id in range(num_items)]
        self.obstacles = [Obstacle(id) for id in range(num_obstacles)]

        self.active_thieves = []
        self.active_items = []

    def within_grid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def surrounded(self, x, y, entity_types):
        if not entity_types:
            return False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            at_x, at_y = x + dx, y + dy
            if self.within_grid(at_x, at_y):
                at_entity = self.grid[at_x][at_y]
                if not any(isinstance(at_entity, entity_type) for entity_type in entity_types):
                    return False
        return True

    def random_position(self, not_surrounded):
        while True:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x][y] is None and not self.surrounded(x, y, not_surrounded):
                return x, y
            
    def reset_game(self):
        self.grid = np.full((self.size, self.size), None)

        for obstacle in self.obstacles:
            x, y = self.random_position(not_surrounded=[])
            self.grid[x][y] = obstacle
            obstacle.set(x, y)

        for item in self.items:
            x, y = self.random_position(not_surrounded=[Obstacle])
            self.grid[x][y] = item
            item.set(x, y)

        for cop in self.cops:
            x, y = self.random_position(not_surrounded=[Obstacle, Item])
            self.grid[x][y] = cop
            cop.set(x, y)

        for thief in self.thieves:
            x, y = self.random_position(not_surrounded=[Obstacle, Cop])
            self.grid[x][y] = thief
            thief.set(x, y)

        self.active_thieves = list(self.thieves)
        self.active_items = list(self.items)
            
    def move_to_delta(self, direction):
        if direction == "stay":
            return 0, 0
        if direction == "up":
            return -1, 0
        if direction == "down":
            return 1, 0
        if direction == "left":
            return 0, -1
        if direction == "right":
            return 0, 1
            
    def is_valid_move(self, agent, direction):
        dx, dy = self.move_to_delta(direction)
        new_x, new_y = agent.x + dx, agent.y + dy
        if not self.within_grid(new_x, new_y):
            return False
        at_entity = self.grid[new_x][new_y]
        return (agent == at_entity or at_entity is None or (isinstance(agent, Thief) and isinstance(at_entity, Item)) or (isinstance(agent, Cop) and isinstance(at_entity, Thief)))
    
    def perform_move(self, agent, direction):
        dx, dy = self.move_to_delta(direction)
        new_x, new_y = agent.x + dx, agent.y + dy
        #print(f"{type(agent).__name__}-{agent.id} moved {direction} from {agent.x, agent.y} to {new_x, new_y}")
        at_entity = self.grid[new_x][new_y]
        if isinstance(agent, Thief) and isinstance(at_entity, Item):
            #print(f"{type(agent).__name__}-{agent.id} caught {type(at_entity).__name__}-{at_entity.id}")
            self.active_items.remove(at_entity)
        if isinstance(agent, Cop) and isinstance(at_entity, Thief):
            #print(f"{type(agent).__name__}-{agent.id} caught {type(at_entity).__name__}-{at_entity.id}")
            self.active_thieves.remove(at_entity)
        self.grid[agent.x][agent.y] = None
        self.grid[new_x][new_y] = agent
        agent.set(new_x, new_y)

    def move_agents(self):
        agents = random.sample(self.active_thieves, len(self.active_thieves)) + random.sample(self.cops, len(self.cops))
        for agent in agents:
            for _ in range(agent.speed):
                excludes = []
                while True:
                    direction, excludes = agent.next_move(excludes)
                    if self.is_valid_move(agent, direction):
                        self.perform_move(agent, direction)
                        break
                    excludes.append(direction)

    def game_over(self):
        return len(self.active_thieves) == 0 or len(self.active_items) == 0
    
    def render(self):
        print("-" * (self.size * 4 + 1))
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                print(f"| {cell if cell is not None else ' '} ", end="")
            print("|")
            print("-" * (self.size * 4 + 1))

