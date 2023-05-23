from agent import Cop, Thief
import numpy as np

def get_distance(agent1, agent2):
    return abs(agent1.x - agent2.x) + abs(agent1.y - agent2.y)

def is_within_view(agent1, agent2, field_of_view):
    return get_distance(agent1, agent2) <= field_of_view

def nearest_agent(env, agent_list, agent):
    min_distance = float('inf')
    nearest_agent = None
    for agent_to_compare in agent_list:
        if is_within_view(agent, agent_to_compare, agent.field_of_view):
            distance = get_distance(agent, agent_to_compare)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = agent_to_compare
    return nearest_agent

def nearest_item(env, agent):
    return nearest_agent(env, env.items, agent)

def nearest_thief(env, agent):
    return nearest_agent(env, env.thieves, agent)

def nearest_cop(env, agent):
    return nearest_agent(env, env.cops, agent)

def nearest_obstacle(env, agent):
    return nearest_agent(env, env.obstacles, agent)

def get_direction(agent1, agent2):
    dx = agent2.x - agent1.x
    dy = agent2.y - agent1.y

    if abs(dx) > abs(dy):
        return (1, 0) if dx > 0 else (-1, 0)
    else:
        return (0, 1) if dy > 0 else (0, -1)

