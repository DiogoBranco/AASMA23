import numpy as np

class Agent:
    def __init__(self, x, y, env, field_of_view):
        self.x = x
        self.y = y
        self.env = env
        self.field_of_view = field_of_view

# agent.py
class QLearningAgent(Agent):
    def __init__(self, x, y, env, field_of_view, num_states, num_actions, alpha=0.5, gamma=0.95, epsilon=0.1):
        super().__init__(x, y, env, field_of_view)
        
        self.Q = np.zeros((num_states, num_actions))  # initialize Q-table
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def choose_action(self, state):
        # ε-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.Q.shape[1])  # explore
        else:
            action = np.argmax(self.Q[state])  # exploit
        return action

    def update(self, state, action, reward, next_state):
        # Q-Learning update
        self.Q[state, action] = self.Q[state, action] + self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        )

class Cop(QLearningAgent):
    def __init__(self, x, y, env, num_states, num_actions, field_of_view=3, alpha=0.5, gamma=0.95, epsilon=0.1):
        super().__init__(x, y, env, field_of_view, num_states, num_actions, alpha, gamma, epsilon)

class Thief(QLearningAgent):
    def __init__(self, x, y, env, num_states, num_actions, field_of_view=3, alpha=0.5, gamma=0.95, epsilon=0.1):
        super().__init__(x, y, env, field_of_view, num_states, num_actions, alpha, gamma, epsilon)
