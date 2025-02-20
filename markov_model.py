import numpy as np

def compute_game_win_probability(p):
    def win_probability(a, b):
        if a >= 4 and (a - b) >= 2:
            return 1.0
        if b >= 4 and (b - a) >= 2:
            return 0.0
        return p * win_probability(a + 1, b) + (1 - p) * win_probability(a, b + 1)
    return win_probability(0, 0)

def compute_set_win_probability(p_game, a=0, b=0, memo=None):
    if memo is None:
        memo = {}
    if (a, b) in memo:
        return memo[(a, b)]
    if a >= 6 and (a - b) >= 2:
        return 1.0
    if b >= 6 and (b - a) >= 2:
        return 0.0
    prob = p_game * compute_set_win_probability(p_game, a + 1, b, memo) + (1 - p_game) * compute_set_win_probability(p_game, a, b + 1, memo)
    memo[(a, b)] = prob
    return prob

def compute_match_win_probability(p_set, sA=0, sB=0, target=2, memo=None):
    if memo is None:
        memo = {}
    if (sA, sB) in memo:
        return memo[(sA, sB)]
    if sA == target:
        return 1.0
    if sB == target:
        return 0.0
    prob = p_set * compute_match_win_probability(p_set, sA + 1, sB, target, memo) + (1 - p_set) * compute_match_win_probability(p_set, sA, sB + 1, target, memo)
    memo[(sA, sB)] = prob
    return prob
