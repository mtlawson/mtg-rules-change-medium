"""
Magic: The Gathering match length simulator

What would happen to match lengths, and by extension tournament lengths,
if we changed who chose the first player of matches 2 and 3?
"""

import numpy as np
import pandas as pd 
import plotnine as plt


def gen_num_games(prob: float):
    """ Returns number of games in the round (either 2 or 3) """
    return np.random.binomial(n=1, p=prob, size=1) + 2


def gen_norm_dist_sum(avg: float, sd: float, size):
    """ Generate sum of normal distribution """
    return sum(np.random.normal(loc=avg, scale=sd, size=size))


def gen_gamma_dist_sum(shape: float, scale: float):
    """ Generate sum of gamma distribution """
    return sum(np.random.gamma(shape=shape, scale=scale, size=2))


def is_blowout(prob: float):
    """ Returns True is round is blowout, False otherwise """
    return bool(np.random.binomial(n=1, p=prob, size=1))


def simulate_match(
    mean_length: float, 
    sd: float, 
    prob_of_three_games: float, 
    is_blowout: bool, 
    gamma_shape: float, 
    gamma_scale: float
):
    """ Simulate the length of a MTG match """

    if is_blowout:
        match_length = gen_gamma_dist_sum(shape=gamma_shape, scale=gamma_scale)
    else:
        num_games = gen_num_games(prob=prob_of_three_games)
        match_length = gen_norm_dist_sum(mean_length, sd, num_games)

    return match_length


def simulate_match_lengths_in_round(
        num_matches_per_round: int,
        average_minutes_per_game: float,
        sd_minutes_per_game: float,
        prob_of_three_games: float,
        prob_of_blowout: float = 0, # default is no blowouts
        blowout_shape_parameter: float = 0,
        blowout_scale_parameter: float = 0,
):
    """ Simulate match lengths for an entire tournament round """

    match_lengths = [simulate_match(average_minutes_per_game, 
                                    sd_minutes_per_game, 
                                    prob_of_three_games,
                                    is_blowout(prob_of_blowout),
                                    blowout_shape_parameter,
                                    blowout_scale_parameter) 
                     for i in range(num_matches_per_round)]

    return match_lengths


def does_round_go_to_time(match_lengths_in_round: list, round_time_limit_minutes: int):
    """ 
    Check all match lengths in the round. If any match exceeds the
    maximum time for that round, then the round has gone to time
    """
    return sum(np.greater_equal(match_lengths_in_round, round_time_limit_minutes)) > 0


def find_prob_of_going_to_time(
        num_rounds_to_simulate: int,
        num_matches_per_round: int,
        average_minutes_per_game: float,
        sd_minutes_per_game: float,
        prob_of_three_games: float,
        round_time_limit_minutes: int,
        prob_of_blowout: float = 0, # default is no blowouts
        blowout_shape_parameter: float = 0,
        blowout_scale_parameter: float = 0
):
    """ 
    Simulate many rounds of gameplay and record how 
    many rounds go to time. The probability of a round
    going to time is the count of rounds that went to time
    divided by the total number of rounds simulated.
    """
        
    went_to_time = [does_round_go_to_time(
                simulate_match_lengths_in_round(
                    num_matches_per_round=num_matches_per_round,
                    prob_of_blowout=prob_of_blowout,
                    blowout_shape_parameter=blowout_shape_parameter,
                    blowout_scale_parameter=blowout_scale_parameter,
                    average_minutes_per_game=average_minutes_per_game,
                    sd_minutes_per_game=sd_minutes_per_game,
                    prob_of_three_games=prob_of_three_games
                ),
                round_time_limit_minutes
            ) for i in range(num_rounds_to_simulate)]

    return sum(went_to_time) / len(went_to_time)


def calc_go_to_time_probs(
        num_rounds_to_simulate: int,
        num_matches_per_round: int,
        sd_minutes_per_game: float,
        prob_of_three_games: float,
        round_time_limit_minutes: int,
        average_minutes_per_game_values: list,
        prob_of_blowout: float = 0, # default is no blowouts
        blowout_shape_parameter: float = 0,
        blowout_scale_parameter: float = 0
) -> list:
    """ 
    Simulate rounds and calculate go to time probabilities
    based on probability of match going to 3 games
    """
    go_to_time_probs = [find_prob_of_going_to_time(
            num_rounds_to_simulate=num_rounds_to_simulate,
            num_matches_per_round=num_matches_per_round,
            round_time_limit_minutes=round_time_limit_minutes,
            prob_of_blowout=prob_of_blowout,
            blowout_shape_parameter=blowout_shape_parameter,
            blowout_scale_parameter=blowout_scale_parameter,
            average_minutes_per_game=average_minutes_per_game_values[i],
            sd_minutes_per_game=sd_minutes_per_game,
            prob_of_three_games=prob_of_three_games)
        for i in range(len(average_minutes_per_game_values))]

    return go_to_time_probs


def density_plot1(num_matches_per_round: int, match_lengths_from_one_round: list):
    """ Density plot for match lengths, new rules, no blowouts, 85 matches/round """

    match_lengths = pd.DataFrame({'Match length': match_lengths_from_one_round})
    (
            plt.ggplot(match_lengths, plt.aes(x='Match length'))
            + plt.geom_density()
            + plt.geom_vline(xintercept=50, color='black', size=2)
            + plt.theme_classic()
            + plt.xlim([0, 55])
    ).save(filename='figures/match_length_density_plot.png')


def density_plot2(num_matches_per_round: int, 
                  match_lengths_from_one_round: list, 
                  match_lengths_from_one_round_with_blowouts: list):
    """ Density plot for match lengths, new rules, blowouts vs. no blowouts, 85 matches/round """

    match_lengths_blowout = pd.DataFrame({
        'Match length': np.concatenate([match_lengths_from_one_round, match_lengths_from_one_round_with_blowouts]),
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


def go_to_time_plot1(go_to_time_probs_new: list, 
                     go_to_time_probs_old: list,
                     average_minutes_per_game_values: list
):
    """ Plot go-to-time probability, new vs. old rules, no blowouts, 85 matches/round """

    time_prob_data = pd.DataFrame({
        'Average minutes per game': np.concatenate([
            average_minutes_per_game_values,
            average_minutes_per_game_values
        ]),
        'P(Go to time)': np.concatenate([
            go_to_time_probs_new,
            go_to_time_probs_old
        ]),
        'Rules': np.concatenate([
            np.repeat('New', len(average_minutes_per_game_values)),
            np.repeat('Old', len(average_minutes_per_game_values))
        ])
    })
    (
        plt.ggplot(time_prob_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
        + plt.geom_line()
        + plt.geom_point()
        + plt.ylim([0, 1])
        + plt.theme_classic()
    ).save(filename='figures/go_to_time_prob_plot.png')


def go_to_time_plot2(go_to_time_probs_new: list, 
          go_to_time_probs_old: list,
          go_to_time_blowout_probs_new: list,
          go_to_time_blowout_probs_old: list,
          average_minutes_per_game_values: list
):
    """ Plot go-to-time probability, new vs. old rules, blowouts vs. no blowouts, 85 matches/round """

    time_prob_blowout_data = pd.DataFrame({
        'Average minutes per game': np.concatenate([
            average_minutes_per_game_values,
            average_minutes_per_game_values,
            average_minutes_per_game_values,
            average_minutes_per_game_values
        ]),
        'P(Go to time)': np.concatenate([
            go_to_time_probs_new,
            go_to_time_probs_old,
            go_to_time_blowout_probs_new,
            go_to_time_blowout_probs_old
        ]),
        'Rules': np.concatenate([
            np.repeat('New, no blowouts', len(average_minutes_per_game_values)),
            np.repeat('Old, no blowouts', len(average_minutes_per_game_values)),
            np.repeat('New, blowouts', len(average_minutes_per_game_values)),
            np.repeat('Old, blowouts', len(average_minutes_per_game_values))
        ])
    })

    (
        plt.ggplot(time_prob_blowout_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
        + plt.geom_line()
        + plt.geom_point()
        + plt.ylim([0, 1])
        + plt.theme_classic()
    ).save(filename='figures/go_to_time_prob_with_blowouts_plot.png')


def go_to_time_plot3(large_go_to_time_probs_new: list, 
                     large_go_to_time_probs_old: list,
                     average_minutes_per_game_values: list
):
    """ Plot go-to-time probability, old vs. new rules, no blowouts, 300 matches/round """

    large_time_prob_data = pd.DataFrame({
        'Average minutes per game': np.concatenate([
            average_minutes_per_game_values,
            average_minutes_per_game_values
        ]),
        'P(Go to time)': np.concatenate([
            large_go_to_time_probs_new,
            large_go_to_time_probs_old
        ]),
        'Rules': np.concatenate([
            np.repeat('New', len(average_minutes_per_game_values)),
            np.repeat('Old', len(average_minutes_per_game_values))
        ])
    })
    (
            plt.ggplot(large_time_prob_data, plt.aes(x='Average minutes per game', y='P(Go to time)', color='Rules'))
            + plt.geom_line()
            + plt.geom_point()
            + plt.ylim([0, 1])
            + plt.theme_classic()
    ).save(filename='figures/go_to_time_300_matches_prob_plot.png')


def main():
    """ Run all simulations and generate figures """

    np.random.seed(23)

    ###############################################################################
    ########## Update these to change the assumptions of the simulations ##########
    ###############################################################################

    NUM_MATCHES_PER_ROUND = 85
    NUM_MATCHES_PER_ROUND_HUGE = 300
    NUM_MINUTES_PER_ROUND = 50
    PROB_WIN_ON_PLAY = 0.55

    CASE_1_AVG_MINUTES_PER_GAME = 13.5
    CASE_1_SD_MINUTES_PER_GAME = 2.5

    CASE_2_BLOWOUT_PROB = 0.10
    CASE_2_LOW_SCALE_PARAM = 2.5
    CASE_2_LOW_SHAPE_PARAM = 2
    CASE_2_HIGH_AVG_MINUTES_PER_GAME = 13.5
    CASE_2_HIGH_SD_MINUTES_PER_GAME = 2.5

    AVERAGE_MINUTES_PER_GAME_VALUES = [12, 12.5, 13, 13.5, 14, 14.5, 15]

    ###############################################################################

    # New rules - Calculate probability of a match having 2 or 3 games
    prob_of_two_games_new = PROB_WIN_ON_PLAY * PROB_WIN_ON_PLAY + (1 - PROB_WIN_ON_PLAY) * PROB_WIN_ON_PLAY
    prob_of_three_games_new = 1 - prob_of_two_games_new

    # Old rules - Calcualte probability of a match having 2 or 3 games
    prob_of_two_games_old = PROB_WIN_ON_PLAY * (1 - PROB_WIN_ON_PLAY) + (1 - PROB_WIN_ON_PLAY) * (1 - PROB_WIN_ON_PLAY)
    prob_of_three_games_old = 1 - prob_of_two_games_old

    ###############################################################################

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
    match_lengths_from_one_round = simulate_match_lengths_in_round(
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        average_minutes_per_game=CASE_1_AVG_MINUTES_PER_GAME,
        sd_minutes_per_game=CASE_1_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_new
    )

    print("Simulating match lengths for a single round - WITH blowouts")
    match_lengths_from_one_round_with_blowouts = simulate_match_lengths_in_round(
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        prob_of_blowout=CASE_2_BLOWOUT_PROB,
        blowout_shape_parameter=CASE_2_LOW_SHAPE_PARAM,
        blowout_scale_parameter=CASE_2_LOW_SCALE_PARAM,
        average_minutes_per_game=CASE_2_HIGH_AVG_MINUTES_PER_GAME,
        sd_minutes_per_game=CASE_2_HIGH_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_new
    )

    print("New rules - simulating rounds to calculate probability of going to time")
    go_to_time_probs_new = calc_go_to_time_probs(
        num_rounds_to_simulate=10000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        sd_minutes_per_game=CASE_1_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_new,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("Old rules - simulating rounds to calculate probability of going to time")
    go_to_time_probs_old = calc_go_to_time_probs(
        num_rounds_to_simulate=10000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        sd_minutes_per_game=CASE_1_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_old,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("New rules with blowouts - simulating rounds to calculate probability of going to time")
    go_to_time_blowout_probs_new = calc_go_to_time_probs(
        num_rounds_to_simulate=10000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        sd_minutes_per_game=CASE_2_HIGH_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_new,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        prob_of_blowout=CASE_2_BLOWOUT_PROB,
        blowout_shape_parameter=CASE_2_LOW_SHAPE_PARAM,
        blowout_scale_parameter=CASE_2_LOW_SCALE_PARAM,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("Old rules with blowouts - simulating rounds to calculate probability of going to time")
    go_to_time_blowout_probs_old = calc_go_to_time_probs(
        num_rounds_to_simulate=10000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND,
        sd_minutes_per_game=CASE_2_HIGH_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_old,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        prob_of_blowout=CASE_2_BLOWOUT_PROB,
        blowout_shape_parameter=CASE_2_LOW_SHAPE_PARAM,
        blowout_scale_parameter=CASE_2_LOW_SCALE_PARAM,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("New rules, large tournament - simulating rounds to calculate probability of going to time")
    large_go_to_time_probs_new = calc_go_to_time_probs(
        num_rounds_to_simulate=1000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND_HUGE,
        sd_minutes_per_game=CASE_1_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_new,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("Old rules, large tournament - simulate rounds to calculate probability of going to time")
    large_go_to_time_probs_old = calc_go_to_time_probs(
        num_rounds_to_simulate=1000,
        num_matches_per_round=NUM_MATCHES_PER_ROUND_HUGE,
        sd_minutes_per_game=CASE_1_SD_MINUTES_PER_GAME,
        prob_of_three_games=prob_of_three_games_old,
        average_minutes_per_game_values=AVERAGE_MINUTES_PER_GAME_VALUES,
        round_time_limit_minutes=NUM_MINUTES_PER_ROUND)

    print("Simulations complete, on to figures!")

    ######################
    ### Make all plots ###
    ######################

    density_plot1(NUM_MATCHES_PER_ROUND, match_lengths_from_one_round)

    density_plot2(NUM_MATCHES_PER_ROUND, match_lengths_from_one_round, match_lengths_from_one_round_with_blowouts)

    go_to_time_plot1(go_to_time_probs_new, go_to_time_probs_old, AVERAGE_MINUTES_PER_GAME_VALUES)

    go_to_time_plot2(go_to_time_probs_new, 
          go_to_time_probs_old,
          go_to_time_blowout_probs_new,
          go_to_time_blowout_probs_old,
          AVERAGE_MINUTES_PER_GAME_VALUES)

    go_to_time_plot3(large_go_to_time_probs_new, large_go_to_time_probs_old, AVERAGE_MINUTES_PER_GAME_VALUES)
    

if __name__ == '__main__':
    main()