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

        self.env = Environment(self.size, self.num_thieves, self.thieves_fov, self.thieves_speed, self.num_cops, self.cops_fov, self.cops_speed, self.num_items, self.num_obstacles, self.model)

    def reset_stats(self):
        self.thieves_wins = 0
        self.cops_wins = 0
        self.draws = 0
        self.aborts = 0
        self.cops_efficiency = []
        self.thieves_efficiency = []
        self.game_steps = []

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

    def perform_game_aux(self, input_flag, render_flag, coop_flag):
        steps = 0
        self.env.reset_game()
        if render_flag:
            self.env.render()
        while not self.env.game_over():
            if input_flag:
                input("Press Enter to perform a new step...")
            if coop_flag:
                self.env.move_agents_coop()
            else:
                self.env.move_agents()
            if render_flag:
                self.env.render()
            steps += 1
            if self.game_steps and steps > 10 * stats.mean(self.game_steps):
                return -1
        return steps
    
    def perform_training_aux(self, coop_flag):
        if self.model == "learning":
            for agent in self.env.thieves + self.env.cops:
                if coop_flag:
                    if len(agent.memory) >= 32:
                        agent.replay_coop(32)
                    else:
                        agent.replay_coop(len(agent.memory))  # Replace with your actual replay method
                else:
                    if len(agent.memory) >= 32:
                        agent.replay(32)
                    else:
                        agent.replay(len(agent.memory)) 

    def perform_game_train(self, input_flag, render_flag, coop_flag):
        steps = self.perform_game_aux(input_flag, render_flag, coop_flag)
        self.perform_training_aux(coop_flag)
        return steps

    def perform_game_test(self, input_flag, render_flag, coop_flag):
        return self.perform_game_aux(input_flag, render_flag, coop_flag)

    def perform_simulation(self, input_flag, render_flag, coop_flag, num_games_train, num_games_test):
        self.reset_stats()
        for id in range(1, num_games_train+1):
            steps = self.perform_game_train(input_flag, render_flag, coop_flag)
            self.game_stats(id, steps)
        print(f"====== [Final Train Results] ======")
        print(f"Thieves Win Ratio: {(self.thieves_wins / num_games_train * 100):.2f}%")
        print(f"Cops Win Ratio: {(self.cops_wins / num_games_train * 100):.2f}%")
        print(f"Draws Ratio: {(self.draws / num_games_train * 100):.2f}%")
        print(f"Aborts Ratio: {(self.aborts / num_games_train * 100):.2f}%")
        print(f"Thieves Avg. Efficiency: {(stats.mean(self.thieves_efficiency)*100):.2f}%")
        print(f"Cops Avg. Efficiency: {(stats.mean(self.cops_efficiency)*100):.2f}%")
        
        input("Press Enter to proceed...")
        
        self.reset_stats()
        for id in range(1, num_games_test+1):
            steps = self.perform_game_test(input_flag, render_flag, coop_flag)
            self.game_stats(id, steps)
        print(f"====== [Final Test Results] ======")
        print(f"Thieves Win Ratio: {(self.thieves_wins / num_games_test * 100):.2f}%")
        print(f"Cops Win Ratio: {(self.cops_wins / num_games_test * 100):.2f}%")
        print(f"Draws Ratio: {(self.draws / num_games_test * 100):.2f}%")
        print(f"Aborts Ratio: {(self.aborts / num_games_test * 100):.2f}%")
        print(f"Thieves Avg. Efficiency: {(stats.mean(self.thieves_efficiency)*100):.2f}%")
        print(f"Cops Avg. Efficiency: {(stats.mean(self.cops_efficiency)*100):.2f}%")
       
        