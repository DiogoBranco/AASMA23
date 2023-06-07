from agent import Agent
from thief import Thief

class Cop(Agent):
    def __init__(self, id, fov, speed, model, env):
        super().__init__(id, fov, speed, model, env)

    def random_move(self, excludes):
        return super().random_move_aux(excludes)
    
    def greedy_move(self, excludes):
        return super().greedy_move_aux(excludes, Thief)

    def learning_move(self, excludes):
        return super().learning_move_aux(excludes)
