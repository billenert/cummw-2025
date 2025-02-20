import numpy as np
from functools import lru_cache

def compute_game_win_probability(p):
    """
    Computes the probability that Player A wins a game,
    given that the probability of winning an individual point is p.
    
    Uses a closed–form approximation that accounts for deuce:
    
      P(game win) = p^4 + 4*p^4*(1-p) + 10*p^4*(1-p)^2 + [20*p^5*(1-p)^3] / (p^2+(1-p)^2)
    
    This formula avoids deep recursion.
    """
    q = 1 - p
    term1 = p**4
    term2 = 4 * p**4 * q
    term3 = 10 * p**4 * (q**2)
    term4 = (20 * p**5 * (q**3)) / (p**2 + q**2)
    return term1 + term2 + term3 + term4

def compute_tiebreak_win_probability(p, max_points=50):
    """
    Computes the probability that Player A wins a tie–break,
    given the probability p of winning an individual point.
    
    A tie–break is played until one player reaches at least 7 points with a 2–point margin.
    This function builds a grid of states (i, j) for points won by A and B respectively,
    using backward dynamic programming. We use a sufficiently high max_points to approximate
    the infinite state space.
    """
    # Create a (max_points+1) x (max_points+1) grid for dp.
    dp = np.zeros((max_points+1, max_points+1))
    
    # Set absorbing states:
    for i in range(max_points+1):
        for j in range(max_points+1):
            if i >= 7 and (i - j) >= 2:
                dp[i, j] = 1.0
            elif j >= 7 and (j - i) >= 2:
                dp[i, j] = 0.0
                
    # Fill the grid in reverse order.
    # We assume that states near the boundary (i == max_points or j == max_points)
    # are almost absorbing (their probability is already set).
    for i in range(max_points-1, -1, -1):
        for j in range(max_points-1, -1, -1):
            # Only update if state (i,j) is not absorbing.
            if not ((i >= 7 and (i-j) >= 2) or (j >= 7 and (j-i) >= 2)):
                dp[i, j] = p * dp[i+1, j] + (1-p) * dp[i, j+1]
    return dp[0, 0]

def compute_set_win_probability(p_game, a=0, b=0, memo=None, p_point=0.65):
    """
    Recursively computes the probability that Player A wins a set,
    given the current game score (a, b) and the game-winning probability p_game.
    
    A set is won if a player reaches at least 6 games and leads by at least 2.
    If the score reaches 6–6, a tie–break is assumed and its win probability is computed
    using the tie–break function based on the point-winning probability p_point.
    """
    if memo is None:
        memo = {}
    if (a, b) in memo:
        return memo[(a, b)]
    if a >= 6 and (a - b) >= 2:
        return 1.0
    if b >= 6 and (b - a) >= 2:
        return 0.0
    if a == 6 and b == 6:
        # Compute tie-break win probability (using point-level probability)
        return compute_tiebreak_win_probability(p_point)
    prob = p_game * compute_set_win_probability(p_game, a+1, b, memo, p_point) \
         + (1 - p_game) * compute_set_win_probability(p_game, a, b+1, memo, p_point)
    memo[(a, b)] = prob
    return prob

def compute_match_win_probability(p_set, sA=0, sB=0, target=2, memo=None):
    """
    Recursively computes the probability that Player A wins the match (best-of-3 sets)
    given the current set score (sA, sB) and the set-winning probability p_set.
    """
    if memo is None:
        memo = {}
    if (sA, sB) in memo:
        return memo[(sA, sB)]
    if sA == target:
        return 1.0
    if sB == target:
        return 0.0
    prob = p_set * compute_match_win_probability(p_set, sA+1, sB, target, memo) \
         + (1 - p_set) * compute_match_win_probability(p_set, sA, sB+1, target, memo)
    memo[(sA, sB)] = prob
    return prob
