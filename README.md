# Cops and Thieves Reinforcement Learning Simulation

## Project Overview

This project implements a multi-agent reinforcement learning environment that simulates a cops and thieves game. In this simulation, cop agents attempt to catch thief agents, while thieves try to collect items without being caught. The project uses deep Q-learning (DQN) to train intelligent agents that interact within a shared grid environment.

## Key Components

### Agents
- **Base Agent Class**: Defines core agent functionality with a Deep Q-Network model, memory replay buffer, and state management.
- **Cop Agents**: Specialized agents that seek to catch thieves, receiving rewards for successful captures.
- **Thief Agents**: Specialized agents that attempt to collect items while avoiding cops, receiving rewards for each item collected.

### Environment
- Grid-based world with configurable size
- Contains agents (cops and thieves), items, and obstacles
- Implements a gym-compatible interface for reinforcement learning
- Handles agent movement, collisions, and reward calculations

### Game Objects
- **Items**: Collectible objects for thieves
- **Obstacles**: Impassable cells on the grid

## Technical Details

### Agent Intelligence Models
The agents can operate in several different intelligence modes:
1. **Random**: Makes random moves
2. **Greedy**: Takes the action that leads toward the closest objective (thief or item)
3. **Reinforcement Learning**: Uses a Deep Q-Network model to learn optimal behavior:
   - Neural network architecture: Sequential model with flattening and dense layers
   - Experience replay buffer for stable learning
   - Epsilon-greedy exploration strategy
   - Field of view for partial observability

### State and Action Spaces
- **State Space**: Local field of view for each agent (partial observability)
- **Action Space**: Discrete set of 4 actions (up, down, left, right)

### Reward System
- Cops: Rewarded for catching thieves
- Thieves: Rewarded for collecting items
- Both: Penalized for invalid moves (e.g., moving into obstacles or out of bounds)

## How to Run

1. **Basic Simulation**: Run the main file to start a simple simulation with default parameters:
   ```
   python main.py
   ```

2. **Customize Parameters**: You can modify the environment parameters in `main.py`:
   - Grid size
   - Number of cops
   - Number of thieves
   - Number of items
   - Number of obstacles
   - Number of episodes

## Output

The simulation will render the environment at each step, showing:
- Cops (C)
- Thieves (T)
- Items (I)
- Obstacles (O)
- Empty spaces ( )

The simulation continues until either all thieves are caught or all items are collected.

## Extending the Project

You can extend this project in several ways:
1. Implement more sophisticated agent intelligence algorithms
2. Add more complex environments with varying reward structures
3. Implement graphical visualization for better usability
4. Add different types of agents, items, or obstacles with special properties
5. Incorporate cooperative behavior among agents of the same type

## Dependencies

- NumPy
- Keras
- TensorFlow
- OpenAI Gym
- Collections (deque)
- Random

## Project Structure

- `agent.py`: Defines the Agent base class and specialized Cop and Thief agents
- `environment.py`: Implements the game environment and core mechanics
- `game_objects.py`: Defines Item and Obstacle classes
- `game.py`: Alternate implementation with simpler game mechanics
- `main.py`: Entry point for running the simulation
- `utils.py`: Helper functions for agents and environment interaction
