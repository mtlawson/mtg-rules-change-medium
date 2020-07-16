""" 
Tests for simulation functions

TODO: Use unit testing framework
"""

import simulate_mtg_rounds as mtg

def test_is_blowout():

    loop_counter = 0

    while loop_counter < 10000:
        assert(mtg.is_blowout(prob=0) is False)
        loop_counter += 1


if __name__ == "__main__":
    test_is_blowout()
