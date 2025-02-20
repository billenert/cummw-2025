import numpy as np

def compute_game_win_probability(p):

    def win_probability(a, b):

        if a >= 4 and (a - b) >= 2:
            return 1.0
        if b >= 4 and (b - a) >= 2:
            return 0.0

        return p * win_probability(a + 1, b) + (1 - p) * win_probability(a, b + 1)
    
    return win_probability(0, 0)
