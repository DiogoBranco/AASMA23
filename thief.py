from agent import Agent
from item import Item

class Thief(Agent):
    def __init__(self, id, x, y, fov, speed, model, env):
        super().__init__(id, x, y, fov, speed, model, env)

    def random_move(self, excludes):
        return super().random_move_aux(excludes)
    
    def greedy_move(self, excludes):
        return super().greedy_move_aux(excludes, Item)
