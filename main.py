from environment import Environment
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Simulation:
    def __init__(self, env, batch_size):
        self.env = env
        self.batch_size = batch_size
        self.max_turns_without_move = 10
        self.num_turns_without_move = 0

    def handle_agents(self, agents, training=True):
        has_moved = False
        total_reward = 0
        done = False
        for agent in agents:
            state = agent.get_agent_view().reshape(agent.dqn_agent.model_input_shape)
            action = agent.act(state)
            new_state, reward, done, info = agent.step(action)
            total_reward += reward
            if new_state is not None:
                new_state = new_state.reshape(agent.dqn_agent.model_input_shape)
                has_moved = info['has_moved']
                if training and len(agent.dqn_agent.memory) > self.batch_size:
                    agent.replay(self.batch_size)
        return has_moved, total_reward, done

    def run_episode(self, num_episodes, training=True):
        for episode in range(num_episodes):
            self.env.reset()
            total_rewards = 0
            while not self.env.is_game_over():
                has_moved_cop, reward_cop, done_cop = self.handle_agents(self.env.cops, training)
                has_moved_thief, reward_thief, done_thief = self.handle_agents(self.env.thieves, training)
                total_rewards += (reward_cop + reward_thief)
                has_moved = has_moved_cop or has_moved_thief
                done = done_cop or done_thief
                if not has_moved:
                    self.num_turns_without_move += 1
                    if self.num_turns_without_move >= self.max_turns_without_move:
                        print(f"No movement for {self.max_turns_without_move} turns. Terminating...")
                        break
                else:
                    self.num_turns_without_move = 0
                if done:
                    print(f"Episode {episode} ended early.")
                    break

                # Render the grid after each round of steps from all agents
                self.env.render()

            print(f"End of {'training' if training else 'evaluation'} episode {episode}. Total reward: {total_rewards}. Remaining thieves: {len(self.env.thieves)}, remaining items: {len(self.env.items)}.")


def main():
    env = Environment(10, 2, 2, 4, 5)
    simulation = Simulation(env, batch_size=32)
    simulation.run_episode(num_episodes=1, training=True)  # Training phase
    simulation.run_episode(num_episodes=1, training=False)  # Evaluation phase

if __name__ == "__main__":
    main()






