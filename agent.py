import random
import numpy as np
from entity import Entity
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent(Entity):
    
    def __init__(self, id, fov, speed, model, env):
        super().__init__(id)
        self.fov = fov
        self.speed = speed
        
        if model == "random":
            self.move = self.random_move
        elif model == "greedy":
            self.move = self.greedy_move
        elif model == "learning":
            self.move = self.learning_move

        # Learning Parameters
        self.state_size = (2*fov + 1) * (2*fov + 1) #len of the vector that contains the grid 
        self.action_size = 5 # number of possible moves (left, up, down, right, stay)

        self.memory = deque(maxlen=200000)
        self.gamma = 0.95 # discount rate
        
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.env = env

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, -1)  # Assuming state is a 2D list, otherwise, no need for this line
            next_state = np.array(next_state).reshape(1, -1)
            target = self.model.predict(state)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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

    def learning_move_aux(self, excludes):
        # Get the current state of the environment grid
        state = self.env.get_state(self)
        state = np.array(state).reshape(1, -1)
        # Use the act method to choose an action
        action_idx = self.act(state)
        # Map the output of the act method to a valid action
        actions = ["up", "down", "left", "right", "stay"]
        chosen_action = actions[action_idx]
        return (chosen_action, excludes)

    def random_move(self, excludes):
        raise NotImplementedError()

    def greedy_move(self, excludes):
        raise NotImplementedError()

    def learning_move(self, excludes):
        raise NotImplementedError()
