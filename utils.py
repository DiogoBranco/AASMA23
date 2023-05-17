from agent import Cop, Thief
import numpy as np

def random_move(env, agent):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        np.random.shuffle(directions)
        for dx, dy in directions:
            new_x = agent.x + dx
            new_y = agent.y + dy
            env.move(agent, new_x, new_y)
            break    

def nearest_item(env, agent):
    min_distance = float('inf')
    nearest_item = None
    for item in env.items:
        distance = abs(agent.x - item.x) + abs(agent.y - item.y)
        if distance <= agent.field_of_view and distance < min_distance:
            min_distance = distance
            nearest_item = item
    return nearest_item

def nearest_thief(env, agent):
    min_distance = float('inf')
    nearest_thief = None
    for thief in env.thieves:
        distance = abs(agent.x - thief.x) + abs(agent.y - thief.y)
        if distance <= agent.field_of_view and distance < min_distance:
            min_distance = distance
            nearest_thief = thief
    return nearest_thief

def greedy_move(env, agent):
    if isinstance(agent, Cop):
        target = nearest_thief(env, agent)
    elif isinstance(agent, Thief):
        target = nearest_item(env, agent)
    else:
        return

    if target is None:
        random_move(env, agent)
    else:
        dx = np.sign(target.x - agent.x)
        dy = np.sign(target.y - agent.y)
        if dx != 0 and not env.move(agent, agent.x + dx, agent.y):
            random_move(env, agent)
        elif dy != 0 and not env.move(agent, agent.x, agent.y + dy):
            random_move(env, agent)