import numpy as np
from markov_model import compute_game_win_probability, compute_set_win_probability, compute_match_win_probability

###############################
# SET-LEVEL LEBERAGE FUNCTIONS
###############################

def compute_set_leberage(prob_win_match_if_win_set, prob_win_match_if_lose_set):
    """
    Equation (4):
      L_i = P(win match | win set i) - P(win match | lose set i).
    """
    return prob_win_match_if_win_set - prob_win_match_if_lose_set

def compute_u_partial_sum(L_list, r_list, alpha):
    """
    Equation (5):
      u_i = sum_{k=1}^i [ (1 - alpha)^(k-1) * L_k * (-1)^(r_k+1) ],
    where r_k = 1 if set k was won, 0 if lost.
    Returns the partial sum u for the sets played so far.
    """
    n = len(L_list)
    sum_val = 0.0
    for k in range(n):
        L_k = L_list[k]
        r_k = r_list[k]
        # (-1)^(r_k+1): +1 if r_k==1 (set won), -1 if r_k==0 (set lost)
        sign_factor = (-1)**(r_k + 1)
        discount = (1 - alpha)**(k)
        sum_val += discount * L_k * sign_factor
    return sum_val

###############################
# FACTOR MATRIX FUNCTIONS
###############################

def build_factor_matrix(points_data):
    """
    Builds the raw factor matrix X_raw (N x 5) for each point.
    The five factors are:
      x1(t): Unforced error in the previous point (0 or 1).
      x2(t): Winner occurrence in the previous point (0 or 1).
      x3(t): Indicator that the current point is a game/set/match point (0 or 1).
      x4(t): Consecutive-points factor (tanh(|CP|^2) with sign).
      x5(t): Consecutive-games factor (tanh(|CG|^2) with sign).
    """
    N = len(points_data)
    X_raw = np.zeros((N, 5), dtype=float)
    consecutive_points = 0
    consecutive_games = 0
    prev_game_winner = None

    for t in range(N):
        # x1 and x2 from previous point (if none, set to 0)
        if t == 0:
            x1 = 0.0
            x2 = 0.0
        else:
            x1 = float(points_data[t-1].get("unforced_error", 0))
            x2 = float(points_data[t-1].get("winner_occurrence", 0))
        # x3: current point is a game/set/match point
        gp = points_data[t].get("game_point", 0)
        sp = points_data[t].get("set_point", 0)
        mp = points_data[t].get("match_point", 0)
        x3 = 1.0 if (gp or sp or mp) else 0.0

        # Update consecutive points factor
        winner = points_data[t]["point_winner"]
        if winner == 1:
            if consecutive_points >= 0:
                consecutive_points += 1
            else:
                consecutive_points = 1
        else:
            if consecutive_points <= 0:
                consecutive_points -= 1
            else:
                consecutive_points = -1
        cp_val = np.tanh((abs(consecutive_points))**2)
        if consecutive_points < 0:
            cp_val = -cp_val
        x4 = cp_val

        # Update consecutive games if this point ends a game
        if points_data[t].get("game_boundary", False):
            if winner == 1:
                if prev_game_winner == 1 or prev_game_winner is None:
                    if consecutive_games >= 0:
                        consecutive_games += 1
                    else:
                        consecutive_games = 1
                else:
                    consecutive_games = 1
                prev_game_winner = 1
            else:
                if prev_game_winner == 2 or prev_game_winner is None:
                    if consecutive_games <= 0:
                        consecutive_games -= 1
                    else:
                        consecutive_games = -1
                else:
                    consecutive_games = -1
                prev_game_winner = 2
        cg_val = np.tanh((abs(consecutive_games))**2)
        if consecutive_games < 0:
            cg_val = -cg_val
        x5 = cg_val

        X_raw[t, :] = [x1, x2, x3, x4, x5]
    return X_raw

def standardize_factor_matrix(X_raw):
    """
    Standardizes each column of the factor matrix (zero mean, unit variance).
    """
    X_std = X_raw.copy()
    n_features = X_raw.shape[1]
    for j in range(n_features):
        col = X_raw[:, j]
        mean_j = np.mean(col)
        std_j = np.std(col)
        if std_j < 1e-9:
            X_std[:, j] = 0.0
        else:
            X_std[:, j] = (col - mean_j) / std_j
    return X_std

###############################
# EXTENDED MOMENTUM CALCULATION
###############################

def calculate_momentum_with_markov(
    points_data,
    alpha=0.2,
    omega=np.array([0.5, 0.2, 0.0, 1.0, -0.1]),  # weight vector for the 5 factors
    p_point=0.65  # probability of winning a point (used to compute p_game)
):
    """
    Computes the cumulative momentum M(t) for each point in points_data.
    
    Steps:
      1. Compute the game-winning probability (p_game) from p_point using a closed-form formula.
      2. Compute the set-winning probability (p_set) from p_game.
      3. Use a match-level Markov chain (best-of-3) to obtain, at each set boundary,
         the conditional match-winning probabilities:
             - prob_win_match_if_win_set and prob_win_match_if_lose_set.
      4. Compute set-level Leberage L_i = P(win match|win set) - P(win match|lose set) and update
         the discounted partial sum u (Equation (5)).
      5. Build and standardize the point-level factor matrix (5 factors).
      6. For each point, the final increment is omega^T * X_std(t) plus the current set-level u.
      7. The cumulative momentum is the running sum of these increments.
    
    points_data: a list of dictionaries, one per point. Each dictionary should include:
         "point_winner": 1 if Player A wins the point, 2 if not.
         "unforced_error", "winner_occurrence", "game_point", "set_point", "match_point" as needed.
         "game_boundary": True if the point ends a game.
         "set_boundary": True if the point ends a set.
         For set boundaries, "set_winner": 1 if Player A wins the set, 0 if lost.
    
    Returns:
         A list of cumulative momentum values.
    """
    # 1. Compute game-winning probability from p_point (using the closed-form formula).
    from markov_model import compute_game_win_probability  # imported already at top
    p_game = compute_game_win_probability(p_point)
    # 2. Compute set-winning probability from p_game.
    p_set = compute_set_win_probability(p_game, 0, 0)
    
    # Initialize match state (for best-of-3): sets won by A and B.
    sA, sB = 0, 0
    
    # Build and standardize the factor matrix for point-level increments.
    X_raw = build_factor_matrix(points_data)
    X_std = standardize_factor_matrix(X_raw)
    
    # Set-level Leberage accumulation: maintain lists for L_i and r_i.
    L_list = []
    r_list = []
    set_leberage_sum = 0.0  # current set-level contribution u (Equation (5))
    
    momentum_values = []
    cumulative_momentum = 0.0
    
    # Process each point.
    for t in range(len(points_data)):
        # Point-level increment: base = omega^T * standardized factor vector.
        inc_base = float(np.dot(omega, X_std[t, :]))
        
        # At a set boundary, compute the conditional match-winning probabilities.
        if points_data[t].get("set_boundary", False):
            # Current match state is (sA, sB).
            # Scenario if Player A wins the set: new state = (sA+1, sB)
            prob_win_if_win = compute_match_win_probability(p_set, sA+1, sB, target=2)
            # Scenario if Player A loses the set: new state = (sA, sB+1)
            prob_win_if_lose = compute_match_win_probability(p_set, sA, sB+1, target=2)
            # Equation (4): L_i = P(win match|win set) - P(win match|lose set)
            L_i = compute_set_leberage(prob_win_if_win, prob_win_if_lose)
            # r_i is provided in the data: 1 if Player A wins the set, 0 if lost.
            r_i = points_data[t].get("set_winner", 0)
            L_list.append(L_i)
            r_list.append(r_i)
            # Equation (5): update u = sum_{k=1}^i [(1 - alpha)^(k-1) * L_k * (-1)^(r_k+1)]
            set_leberage_sum = compute_u_partial_sum(L_list, r_list, alpha)
            # Update match state based on actual outcome.
            if r_i == 1:
                sA += 1
            else:
                sB += 1
        
        # Final increment for this point: base increment plus current set-level contribution.
        inc_final = inc_base + set_leberage_sum
        cumulative_momentum += inc_final
        momentum_values.append(cumulative_momentum)
    
    return momentum_values

###############################
# EXAMPLE USAGE
###############################

if __name__ == "__main__":
    # A toy example of 8 points spanning 2 sets (best-of-3 match).
    example_points_data = [
        # Set 1, Game 1, Point 1
        dict(point_winner=1, unforced_error=0, winner_occurrence=1,
             game_point=0, set_point=0, match_point=0,
             game_boundary=False, set_boundary=False),
        # Set 1, Game 1, Point 2 (ends game)
        dict(point_winner=1, unforced_error=0, winner_occurrence=0,
             game_point=1, set_point=0, match_point=0,
             game_boundary=True, set_boundary=False),
        # Set 1, Game 2, Point 1
        dict(point_winner=2, unforced_error=1, winner_occurrence=0,
             game_point=0, set_point=0, match_point=0,
             game_boundary=False, set_boundary=False),
        # Set 1, Game 2, Point 2 (ends game & set; assume Player A wins Set 1)
        dict(point_winner=1, unforced_error=0, winner_occurrence=0,
             game_point=1, set_point=1, match_point=0,
             game_boundary=True, set_boundary=True,
             set_winner=1),
        # Set 2, Game 1, Point 1
        dict(point_winner=2, unforced_error=0, winner_occurrence=1,
             game_point=0, set_point=0, match_point=0,
             game_boundary=False, set_boundary=False),
        # Set 2, Game 1, Point 2 (ends game)
        dict(point_winner=2, unforced_error=1, winner_occurrence=0,
             game_point=1, set_point=0, match_point=0,
             game_boundary=True, set_boundary=False),
        # Set 2, Game 2, Point 1
        dict(point_winner=2, unforced_error=0, winner_occurrence=0,
             game_point=0, set_point=0, match_point=0,
             game_boundary=False, set_boundary=False),
        # Set 2, Game 2, Point 2 (ends game & set; assume Player A loses Set 2)
        dict(point_winner=2, unforced_error=0, winner_occurrence=1,
             game_point=1, set_point=1, match_point=0,
             game_boundary=True, set_boundary=True,
             set_winner=0),
    ]
    
    # Use a point-winning probability of 0.65.
    momentum_vals = calculate_momentum_with_markov(example_points_data, alpha=0.2, p_point=0.65)
    
    print("Cumulative momentum values at each point:")
    for i, m in enumerate(momentum_vals):
        print(f" Point {i}: M = {m:.4f}")