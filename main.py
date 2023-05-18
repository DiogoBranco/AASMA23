from environment import Environment
from utils import greedy_move

def main():

    num_states = 10000  # adjust these values based on your problem
    num_actions = 4

    env = Environment(10, 2, 3, 4, 5)

    max_turns_without_move = 10
    num_turns_without_move = 0

    while not env.is_game_over():
        has_moved = False
        for cop in env.cops:
            if greedy_move(env, cop):
                has_moved = True
        for thief in env.thieves:
            if greedy_move(env, thief):
                has_moved = True
        if not has_moved:
            num_turns_without_move += 1
            if num_turns_without_move >= max_turns_without_move:
                print("No movement for {} turns. Terminating...".format(max_turns_without_move))
                break
        else:
            num_turns_without_move = 0
        env.render()
        print("\n" + "="*10)

    print("Game Over. Remaining thieves: {}, remaining items: {}.".format(len(env.thieves), len(env.items)))

if __name__ == "__main__":
    main()

