import statistics as stats
from environment import Environment

class Simulation:

    def __init__(self, size, num_thieves, thieves_fov, thieves_speed, num_cops, cops_fov, cops_speed, num_items, num_obstacles, model):
        self.size = size

        self.num_thieves = num_thieves
        self.thieves_fov = thieves_fov
        self.thieves_speed = thieves_speed

        self.num_cops = num_cops
        self.cops_fov = cops_fov
        self.cops_speed = cops_speed

        self.num_items = num_items
        self.num_obstacles = num_obstacles

        self.model = model

        self.thieves_wins = 0
        self.cops_wins = 0
        self.draws = 0
        self.aborts = 0
        self.cops_efficiency = []
        self.thieves_efficiency = []
        self.game_steps = []

        self.env = Environment(self.size, self.num_thieves, self.thieves_fov, self.thieves_speed, self.num_cops, self.cops_fov, self.cops_speed, self.num_items, self.num_obstacles, self.model)

    def game_stats(self, id, steps):
        if steps == -1:
            print(f"=== [Game {id}] Aborted! ===")
            self.aborts += 1
            return
        num_active_thieves = len(self.env.active_thieves)
        num_active_items = len(self.env.active_items)
        cops_efficiency = (self.num_thieves - num_active_thieves) / self.num_thieves
        thieves_efficiency = (self.num_items - num_active_items) / self.num_items
        self.game_steps.append(steps)
        self.cops_efficiency.append(cops_efficiency)
        self.thieves_efficiency.append(thieves_efficiency)
        if num_active_items == 0 and num_active_thieves > 0:
            self.thieves_wins += 1
            print(f"=== [Game {id}] Thieves Win! ===")
        elif num_active_items > 0 and num_active_thieves == 0:
            self.cops_wins += 1
            print(f"=== [Game {id}] Cops Win! ===")
        else:
            self.draws += 1
            print(f"=== [Game {id}] Draw! ===")
        #print(f"Thieves Efficiency: {thieves_efficiency}")
        #print(f"Cops Efficiency {cops_efficiency}")

    def perform_game(self, input_flag, render_flag):
        steps = 0
        self.env.reset_game()
        if render_flag:
            self.env.render()
        while not self.env.game_over():
            if input_flag:
                input("Press Enter to perform a new step...")
            self.env.move_agents()
            if render_flag:
                self.env.render()
            steps += 1
            if self.game_steps and steps > 10 * stats.mean(self.game_steps):
                return -1
        return steps

    def perform_simulation(self, input_flag, render_flag, num_games):
        for id in range(1, num_games+1):
            steps = self.perform_game(input_flag, render_flag)
            self.game_stats(id, steps)
        print(f"====== [Final Results] ======")
        print(f"Thieves Win Ratio: {(self.thieves_wins / num_games * 100):.2f}%")
        print(f"Cops Win Ratio: {(self.cops_wins / num_games * 100):.2f}%")
        print(f"Draws Ratio: {(self.draws / num_games * 100):.2f}%")
        print(f"Aborts Ratio: {(self.aborts / num_games * 100):.2f}%")
        print(f"Thieves Avg. Efficiency: {(stats.mean(self.thieves_efficiency)*100):.2f}%")
        print(f"Cops Avg. Efficiency: {(stats.mean(self.cops_efficiency)*100):.2f}%")
