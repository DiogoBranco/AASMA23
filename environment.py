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
        self.grid = np.full((size, size), None)

        self.thieves = []
        self.cops = []
        self.items = []
        self.obstacles = []

        for id in range(num_thieves):
            x, y = self.place_randomly()
            thief = Thief(id, x, y, thieves_fov, thieves_speed, model, self)
            self.grid[x][y] = thief
            self.thieves.append(thief)

        for id in range(num_cops):
            x, y = self.place_randomly()
            cop = Cop(id, x, y, cops_fov, cops_speed, model, self)
            self.grid[x][y] = cop
            self.cops.append(cop)

        for id in range(num_items):
            x, y = self.place_randomly()
            item = Item(id, x, y)
            self.grid[x][y] = item
            self.items.append(item)

        for id in range(num_obstacles):
            x, y = self.place_randomly()
            obstacle = Obstacle(id, x, y)
            self.grid[x][y] = obstacle
            self.obstacles.append(obstacle)

    def place_randomly(self):
        while True:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x][y] is None:
                return x, y
            
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
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return False
        at_entity = self.grid[new_x][new_y]
        return (agent == at_entity or at_entity is None or (isinstance(agent, Thief) and isinstance(at_entity, Item)) or (isinstance(agent, Cop) and isinstance(at_entity, Thief)))
    
    def perform_move(self, agent, direction):
        dx, dy = self.move_to_delta(direction)
        new_x, new_y = agent.x + dx, agent.y + dy
        print(f"{type(agent).__name__}-{agent.id} moved {direction} from {agent.x, agent.y} to {new_x, new_y}")
        at_entity = self.grid[new_x][new_y]
        if isinstance(agent, Thief) and isinstance(at_entity, Item):
            print(f"{type(agent).__name__}-{agent.id} caught {type(at_entity).__name__}-{at_entity.id}")
            self.items.remove(at_entity)
        if isinstance(agent, Cop) and isinstance(at_entity, Thief):
            print(f"{type(agent).__name__}-{agent.id} caught {type(at_entity).__name__}-{at_entity.id}")
            self.thieves.remove(at_entity)
        self.grid[agent.x][agent.y] = None
        agent.x, agent.y = new_x, new_y
        self.grid[new_x][new_y] = agent

    def step(self, input_flag):
        if input_flag:
            input("Press Enter to proceed to next step...")

        for thief in random.sample(self.thieves, len(self.thieves)):
            for _ in range(thief.speed):
                excludes = []
                while True:
                    direction, excludes = thief.next_move(excludes)
                    if self.is_valid_move(thief, direction):
                        self.perform_move(thief, direction)
                        break
                    excludes.append(direction)

        for cop in random.sample(self.cops, len(self.cops)):
            for _ in range(cop.speed):
                excludes = []
                while True:
                    direction, excludes = cop.next_move(excludes)
                    if self.is_valid_move(cop, direction):
                        self.perform_move(cop, direction)
                        break
                    excludes.append(direction)
            
    def is_game_over(self):
        if len(self.thieves) == 0:
            print("Cops won the game!")
            return True
        if len(self.items) == 0:
            print("Thieves won the game!")
            return True
        return False

    def render(self):
        print("-" * (self.size * 4 + 1))
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                print(f"| {cell if cell is not None else ' '} ", end="")
            print("|")
            print("-" * (self.size * 4 + 1))
