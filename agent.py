import random
from entity import Entity

class Agent(Entity):
    
    def __init__(self, id, x, y, fov, speed, model, env):
        super().__init__(id, x, y)
        self.fov = fov
        self.speed = speed
        
        if model == "random":
            self.move = self.random_move
        elif model == "greedy":
            self.move = self.greedy_move
        elif model == "learning":
            self.move = self.learning_move

        self.env = env

    def next_move(self, excludes):
        return self.move(excludes)
        
    def entities_in_fov(self, entity_type):
        visible_entities = []
        for dx in range(-self.fov, self.fov+1):
            for dy in range(-self.fov, self.fov+1):
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < self.env.size and 0 <= new_y < self.env.size:
                    at_entity = self.env.grid[new_x][new_y]
                    if isinstance(at_entity, entity_type):
                        visible_entities.append(at_entity)
        return visible_entities

    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    def random_move_aux(self, excludes):
        valid_moves = list(set(["up", "down", "left", "right"]) - set(excludes))
        return (random.choice(valid_moves), excludes) if valid_moves else ("stay", excludes)
    
    def greedy_move_aux(self, excludes, chasing_entity):
        greedy_moves = []
        visible_entities = self.entities_in_fov(chasing_entity)
        if visible_entities:
            closest_thief = min(visible_entities, key=lambda thief: self.manhattan_distance(self.x, self.y, thief.x, thief.y))
            dx, dy = closest_thief.x - self.x, closest_thief.y - self.y
            if dx > 0:
                greedy_moves.append("down")
            elif dx < 0:
                greedy_moves.append("up")
            if dy > 0:
                greedy_moves.append("right")
            elif dy < 0:
                greedy_moves.append("left")
        valid_moves = list(set(greedy_moves) - set(excludes)) or list(set(["up", "down", "left", "right"]) - set(excludes))
        return (random.choice(valid_moves), excludes) if valid_moves else ("stay", excludes)

    def random_move(self, excludes):
        raise NotImplementedError()

    def greedy_move(self, excludes):
        raise NotImplementedError()

    def learning_move(self, excludes):
        raise NotImplementedError()
