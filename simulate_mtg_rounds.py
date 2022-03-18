"""
Magic: The Gathering match length simulator

What would happen to match lengths, and by extension tournament lengths,
if we changed who chose the first player of matches 2 and 3?

What would happen if we used an unnecessary amount of OOP techniques
to create the simulation code?????
"""


import numpy as np
import pandas as pd 
import plotnine as plt

# Module Constants
PROB_WIN_ON_PLAY = 0.55
MAX_MINUTES_PER_ROUND = 50
NUM_MATCHES_PER_ROUND = 85
NUM_MATCHES_PER_ROUND_HUGE = 300
NUM_ROUNDS_SIMULATE = 10000
NUM_ROUNDS_SIMULATE_HUGE = 1000

AVG_MINUTES_PER_GAME = 13.5
SD_MINUTES_PER_GAME = 2.5
BLOWOUT_PROB = 0.10
GAMMA_SCALE_PARAM = 2.5
GAMMA_SHAPE_PARAM = 2

AVERAGE_MINUTES_PER_GAME_LIST = [12, 12.5, 13, 13.5, 14, 14.5, 15]


def gen_num_games(prob: float) -> int:
    """ Returns number of games in the round (either 2 or 3) """
    return np.random.binomial(n=1, p=prob, size=1) + 2


def gen_norm_dist_sum(avg: float, sd: float, size) -> float:
    """ Generate sum of normal distribution """
    return sum(np.random.normal(loc=avg, scale=sd, size=size))


def gen_gamma_dist_sum(shape: float, scale: float) -> float:
    """ Generate sum of gamma distribution """
    return sum(np.random.gamma(shape=shape, scale=scale, size=2))


def check_is_blowout(prob: float) -> bool:
    """ Returns True is round is blowout, False otherwise """
    return bool(np.random.binomial(n=1, p=prob, size=1))

class MagicSimulator():

    def __init__(self,
        num_rounds_to_simulate: int, 
        num_matches_per_round: int,
        avg_minutes_per_game: float,
        sd_minutes_per_game: float,
        
        # default is no blowouts
        prob_of_blowout: float,
        gamma_scale_param: float,
        gamma_shape_param: float
        ):

        self.num_rounds_to_simulate = num_rounds_to_simulate
        self.num_matches_per_round = num_matches_per_round
        self.avg_minutes_per_game = avg_minutes_per_game
        self.sd_minutes_per_game = sd_minutes_per_game      
        self.prob_of_blowout = prob_of_blowout
        self.gamma_scale_param = gamma_scale_param
        self.gamma_shape_param = gamma_shape_param


    def calc_prob_three_games(self, use_new_rules: bool) -> float:
        """ Calculate probability of game going to 3 games given a set of rules"""

        # New rules - Calculate probability of a match having 2 or 3 games
        if use_new_rules:     
            prob_of_two_games = PROB_WIN_ON_PLAY * PROB_WIN_ON_PLAY + (1 - PROB_WIN_ON_PLAY) * PROB_WIN_ON_PLAY 
       
        # Old rules - Calculate probability of a match having 2 or 3 games
        else:
            prob_of_two_games = PROB_WIN_ON_PLAY * (1 - PROB_WIN_ON_PLAY) + (1 - PROB_WIN_ON_PLAY) * (1 - PROB_WIN_ON_PLAY)
        
        prob_of_three_games = 1 - prob_of_two_games

        return prob_of_three_games

    
    def simulate_match(self, use_new_rules: bool, blowouts: bool) -> float:
        """ Simulate the length of a MTG match """

        prob_of_three_games = self.calc_prob_three_games(use_new_rules)

        if blowouts:
            is_blowout = check_is_blowout(self.prob_of_blowout)
        else:
            is_blowout = False

        if is_blowout:
            match_length = gen_gamma_dist_sum(
                shape=self.gamma_shape_param, 
                scale=self.gamma_scale_param)
        else:
            num_games = gen_num_games(prob=prob_of_three_games)
            match_length = gen_norm_dist_sum(
                self.avg_minutes_per_game, 
                self.sd_minutes_per_game, 
                num_games)

        return match_length

    
    def simulate_round(self, use_new_rules: bool, blowouts: bool) -> tuple:
        """ Simulate match lengths for an entire tournament round """

        round_went_to_time = False
        match_lengths = []

        for i in range(self.num_matches_per_round):
            match_length = self.simulate_match(use_new_rules, blowouts)
            match_lengths.append(match_length)

            # Check if indicator already set
            if round_went_to_time:
                pass
            else:
                # Check if round went to time
                if match_length >= MAX_MINUTES_PER_ROUND:
                    round_went_to_time = True

        return (match_lengths, round_went_to_time)

       
    def calc_prob_of_going_to_time(self, use_new_rules: bool, blowouts: bool) -> float:
        """ 
        Simulate many rounds of gameplay and record how 
        many rounds go to time. The probability of a round
        going to time is the count of rounds that went to time
        divided by the total number of rounds simulated.
        """
            
        count_went_to_time = 0

        for i in range(self.num_rounds_to_simulate):
            match_lengths, went_to_time = self.simulate_round(use_new_rules, blowouts)
            if went_to_time:
                count_went_to_time += 1

        return count_went_to_time / self.num_rounds_to_simulate

class MagicSimRunner():

    def __init__(self):
        self.simulations_complete = False
        self.mtg_sim = None
        self.match_lengths_from_one_round = None
        self.match_lengths_from_one_round_with_blowouts = None
        self.go_to_time_probs_new = None
        self.go_to_time_probs_old = None
        self.go_to_time_blowout_probs_new = None
        self.go_to_time_blowout_probs_old = None
        self.large_go_to_time_probs_new = None
        self.large_go_to_time_probs_old = None


    def calc_go_to_time_probs(self, num_rounds: int, num_matches: int, use_new_rules: bool, blowouts: bool) -> list:
        """ 
        Simulate rounds and calculate go to time probabilities
        based on probability of match going to 3 games
        """
        go_to_time_probs = []
        
        for avg in AVERAGE_MINUTES_PER_GAME_LIST:
            mtg_sim = MagicSimulator(num_rounds, 
                                     num_matches, 
                                     avg, 
                                     SD_MINUTES_PER_GAME, 
                                     BLOWOUT_PROB, 
                                     GAMMA_SCALE_PARAM,
                                     GAMMA_SHAPE_PARAM)

            prob_go_to_time = mtg_sim.calc_prob_of_going_to_time(use_new_rules, blowouts)
            go_to_time_probs.append(prob_go_to_time)

        return go_to_time_probs


    def density_plot1(self):
        """ Density plot for match lengths, new rules, no blowouts, 85 matches/round """

        match_lengths = pd.DataFrame({'Match length': self.match_lengths_from_one_round})
        (
                plt.ggplot(match_lengths, plt.aes(x='Match length'))
                + plt.geom_density()
                + plt.geom_vline(xintercept=50, color='black', size=2)
                + plt.theme_classic()
                + plt.xlim([0, 55])
        ).save(filename='figures/match_length_density_plot.png')


    def density_plot2(self):
        """ Density plot for match lengths, new rules, blowouts vs. no blowouts, 85 matches/round """

        num_matches_per_round = len(self.match_lengths_from_one_round)

        match_lengths_blowout = pd.DataFrame({
            'Match length': np.concatenate([self.match_lengths_from_one_round, 
                                            self.match_lengths_from_one_round_with_blowouts]),
            'Blowouts': np.concatenate([
                np.repeat('No', num_matches_per_round),
                np.repeat('Yes', num_matches_per_round)
            ])
        })
        (
            plt.ggplot(match_lengths_blowout, plt.aes(x='Match length', color='Blowouts'))
            + plt.geom_density()
            + plt.geom_vline(xintercept=50, color='black', size=2)
            + plt.xlim([0, 55])
            + plt.theme_classic()
        ).save(filename='figures/match_length_with_blowout_density_plot.png')


    def go_to_time_plot1(self):
        """ Plot go-to-time probability, new vs. old rules, no blowouts, 85 matches/round """

        time_prob_data = pd.DataFrame({
            'Average minutes per game': np.concatenate([
                AVERAGE_MINUTES_PER_GAME_LIST,
                AVERAGE_MINUTES_PER_GAME_LIST
            ]),
            'P(Go to time)': np.concatenate([
                self.go_to_time_probs_new,
                self.go_to_time_probs_old
            ]),
            'Rules': np.concatenate([
                np.repeat('New', len(AVERAGE_MINUTES_PER_GAME_LIST)),
                np.repeat('Old', len(AVERAGE_MINUTES_PER_GAME_LIST))
            ])
        })
        (
            plt.ggplot(time_prob_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
            + plt.geom_line()
            + plt.geom_point()
            + plt.ylim([0, 1])
            + plt.theme_classic()
        ).save(filename='figures/go_to_time_prob_plot.png')


    def go_to_time_plot2(self):
        """ Plot go-to-time probability, new vs. old rules, blowouts vs. no blowouts, 85 matches/round """

        time_prob_blowout_data = pd.DataFrame({
            'Average minutes per game': np.concatenate([
                AVERAGE_MINUTES_PER_GAME_LIST,
                AVERAGE_MINUTES_PER_GAME_LIST,
                AVERAGE_MINUTES_PER_GAME_LIST,
                AVERAGE_MINUTES_PER_GAME_LIST
            ]),
            'P(Go to time)': np.concatenate([
                self.go_to_time_probs_new,
                self.go_to_time_probs_old,
                self.go_to_time_blowout_probs_new,
                self.go_to_time_blowout_probs_old
            ]),
            'Rules': np.concatenate([
                np.repeat('New, no blowouts', len(AVERAGE_MINUTES_PER_GAME_LIST)),
                np.repeat('Old, no blowouts', len(AVERAGE_MINUTES_PER_GAME_LIST)),
                np.repeat('New, blowouts', len(AVERAGE_MINUTES_PER_GAME_LIST)),
                np.repeat('Old, blowouts', len(AVERAGE_MINUTES_PER_GAME_LIST))
            ])
        })

        (
            plt.ggplot(time_prob_blowout_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
            + plt.geom_line()
            + plt.geom_point()
            + plt.ylim([0, 1])
            + plt.theme_classic()
        ).save(filename='figures/go_to_time_prob_with_blowouts_plot.png')


    def go_to_time_plot3(self):
        """ Plot go-to-time probability, old vs. new rules, no blowouts, 300 matches/round """

        large_time_prob_data = pd.DataFrame({
            'Average minutes per game': np.concatenate([
                AVERAGE_MINUTES_PER_GAME_LIST,
                AVERAGE_MINUTES_PER_GAME_LIST
            ]),
            'P(Go to time)': np.concatenate([
                self.large_go_to_time_probs_new,
                self.large_go_to_time_probs_old
            ]),
            'Rules': np.concatenate([
                np.repeat('New', len(AVERAGE_MINUTES_PER_GAME_LIST)),
                np.repeat('Old', len(AVERAGE_MINUTES_PER_GAME_LIST))
            ])
        })
        (
                plt.ggplot(large_time_prob_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
                + plt.geom_line()
                + plt.geom_point()
                + plt.ylim([0, 1])
                + plt.theme_classic()
        ).save(filename=f'figures/go_to_time_{NUM_MATCHES_PER_ROUND_HUGE}_matches_prob_plot.png')


    def run_simulations(self):
        """ Run all simulations and generate figures """

        np.random.seed(23)

        self.mtg_sim = MagicSimulator(NUM_ROUNDS_SIMULATE, 
                                      NUM_MATCHES_PER_ROUND, 
                                      AVG_MINUTES_PER_GAME, 
                                      SD_MINUTES_PER_GAME, 
                                      BLOWOUT_PROB, 
                                      GAMMA_SCALE_PARAM,
                                      GAMMA_SHAPE_PARAM)

          ##################### 
         #######################
        ### START SIMULATIONS ###
         #######################
          #####################

        print("Magic: The Gathering match length simulator")
        print("")
        print("Patience! Some of these simulations might take a bit to run, depending on your hardware")
        print("")
    
        print("Simulating match lengths for a single round - no blowouts")
        self.match_lengths_from_one_round = self.mtg_sim.simulate_round(True, False)[0]

        print("Simulating match lengths for a single round - WITH blowouts")
        self.match_lengths_from_one_round_with_blowouts = self.mtg_sim.simulate_round(True, True)[0]

        print("New rules - simulating rounds to calculate probability of going to time")
        self.go_to_time_probs_new = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE, NUM_MATCHES_PER_ROUND, True, False)

        print("Old rules - simulating rounds to calculate probability of going to time")
        self.go_to_time_probs_old = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE, NUM_MATCHES_PER_ROUND, False, False)

        print("New rules with blowouts - simulating rounds to calculate probability of going to time")
        self.go_to_time_blowout_probs_new = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE, NUM_MATCHES_PER_ROUND, True, True)

        print("Old rules with blowouts - simulating rounds to calculate probability of going to time")
        self.go_to_time_blowout_probs_old = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE, NUM_MATCHES_PER_ROUND, False, True)

        print("New rules, large tournament - simulating rounds to calculate probability of going to time")
        self.large_go_to_time_probs_new = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE_HUGE, NUM_MATCHES_PER_ROUND_HUGE, True, False)

        print("Old rules, large tournament - simulate rounds to calculate probability of going to time")
        self.large_go_to_time_probs_old = self.calc_go_to_time_probs(NUM_ROUNDS_SIMULATE_HUGE, NUM_MATCHES_PER_ROUND_HUGE, False, False)

        print("Simulations complete, on to figures!")
        self.simulations_complete = True


    def make_all_plots(self):
        """ Make all of the plots"""

        if self.simulations_complete:
            self.density_plot1()
            self.density_plot2()
            self.go_to_time_plot1()
            self.go_to_time_plot2()
            self.go_to_time_plot3()
        else:
            print("Run simulations first!!!!")


if __name__ == "__main__":
    mtg_sim_runner = MagicSimRunner()
    mtg_sim_runner.run_simulations()
    mtg_sim_runner.make_all_plots()

    

    
