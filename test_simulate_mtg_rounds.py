""" 
Tests for simulation functions

TODO: Use unit testing framework
"""

import simulate_mtg_rounds as mtg

def test_gen_num_games():
    pass


def test_gen_norm_dist_sum():
    pass


def test_gen_gamma_dist_sum():
    pass


def test_is_blowout():

    loop_counter = 0

    while loop_counter < 10000:
        assert(mtg.is_blowout(prob=0) is False)
        loop_counter += 1
        

def test_simulate_match():
    pass


def test_simulate_match_lengths_in_round():
    pass


def test_does_round_go_to_time(match_lengths_in_round: list):
    pass


def test_find_prob_of_going_to_time():
    pass


if __name__ == "__main__":
    test_is_blowout()
