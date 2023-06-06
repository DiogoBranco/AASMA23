import random
import argparse
from environment import Environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-size", help="Size of the game environment grid", type=int, default=4)
    parser.add_argument("-thieves", help="Number of thieves in the game", type=int, default=2)
    parser.add_argument("-thieves-fov", help="Field of view of thieves in the game", type=int, default=1)
    parser.add_argument("-thieves-speed", help="Speed of movement of thieves in the game", type=int, default=1)
    parser.add_argument("-cops", help="Number of cops in the game", type=int, default=2)
    parser.add_argument("-cops-fov", help="Field of view of cops in the game", type=int, default=1)
    parser.add_argument("-cops-speed", help="Speed of movement of cops in the game", type=int, default=1)
    parser.add_argument("-obstacles", help="Number of obstacles in the game", type=int, default=3)
    parser.add_argument("-items", help="Number of items in the game", type=int, default=2)
    parser.add_argument("-model", help="Intelligence model used by the agents", type=str, choices=["random", "greedy", "learning"], required=True)
    parser.add_argument("-seed", help="The seed for the random number generator", type=int, default=None)
    parser.add_argument("-input", help="Require input for every step", action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    environment = Environment(args.size, args.thieves, args.thieves_fov, args.thieves_speed, args.cops, args.cops_fov, args.cops_speed, args.items, args.obstacles, args.model)

    environment.render()
    while not environment.is_game_over():
        environment.step(args.input)
        environment.render()
