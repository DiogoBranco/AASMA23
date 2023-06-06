from environment import Environment

def main():
    env = Environment(4, 1, 1, 1, 1)

    max_turns_without_move = 10
    num_turns_without_move = 0
    num_episodes = 1  # number of episodes for training
    batch_size = 32   # mini-batch size for replay

    for episode in range(num_episodes):

        env.render()
        while not env.is_game_over():
            env.step()  # call the step method of the Environment class         
            env.render()
            print("\n" + "="*4)

        print(f"End of episode {episode}. Remaining thieves: {len(env.thieves)}, remaining items: {len(env.items)}.")

if __name__ == "__main__":
    main()



