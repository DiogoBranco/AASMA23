from environment import Environment

def main():
    env = Environment(10, 2, 2, 4, 5)

    max_turns_without_move = 10
    num_turns_without_move = 0
    num_episodes = 1  # number of episodes for training
    batch_size = 32   # mini-batch size for replay

    def handle_agents(agents):
        has_moved = False
        for agent in agents:
            state = agent.get_state()
            action = agent.choose_action(state)
            new_state, reward, done, _ = agent.step(action)
            if new_state is not None:
                has_moved = True
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
        return has_moved

    for episode in range(num_episodes):
        env.reset()  # if you have implemented a reset method in your environment

        while not env.is_game_over():
            has_moved = handle_agents(env.cops) or handle_agents(env.thieves)
            if not has_moved:
                num_turns_without_move += 1
                if num_turns_without_move >= max_turns_without_move:
                    print(f"No movement for {max_turns_without_move} turns. Terminating...")
                    break
            else:
                num_turns_without_move = 0
            env.render()
            print("\n" + "="*10)

        print(f"End of episode {episode}. Remaining thieves: {len(env.thieves)}, remaining items: {len(env.items)}.")

if __name__ == "__main__":
    main()



