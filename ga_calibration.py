import params
import random as rand
import numpy as np
import bio_model

# =========================================================================
# Genetic Algorithm – Calibration Our Model for Min Error
# =========================================================================

PARAM_NAMES = [
    "m",
    "n",
    "b",
    "k_min",
    "j",
    "T_opt",
    "h",
    "a"
]

PARAM_RANGES = {
    "m":     (0.8,   0.9),
    "n":     (0.901, 1.1),
    "b":     (0.1,   0.5),
    "k_min": (0.0005, 0.002),
    "j":     (0.01,  0.05),
    "T_opt": (26.0,  30.0),
    "h":     (0.4,   0.8),
    "a":     (0.2,   0.7)
}

PARAM_STD_SCALE = {
    "m":     0.05,
    "n":     0.05,
    "b":     0.05,
    "k_min": 0.0005,
    "j":     0.005,
    "T_opt": 0.5,
    "h":     0.05,
    "a":     0.05
}

PARAM_RATE_SCALE = {
    "m":     0.4,
    "n":     0.4,
    "b":     0.4,
    "k_min": 0.4,
    "j":     0.4,
    "T_opt": 0.4,
    "h":     0.4,
    "a":     0.4
}

def generate_genome():
    '''
    Generate a random value for each of the params
    '''
    genome = []
    for name in PARAM_NAMES:
        low, high = PARAM_RANGES[name]
        genome.append(rand.uniform(low, high))
    return genome


def generate_population(size, seed_ind=None):
    """Generate a population of candidate parameter tuples."""
    population = []
    if seed_ind is not None:
        population.append(seed_ind[:])
    while len(population) < size:
        population.append(generate_genome())
    return population


def tank_error(Feed_arr, T_arr, Data_days, Data, N_fish, DO_const, Q_const):
    """
    Compute calibration error ONLY on actual measured days, with index safety.
    """
    initial_weight = Data[0] / 1000.0   # kg
    days_full = np.arange(197)  

    meas_full = np.interp(days_full, Data_days, Data)
    W_est = meas_full / 1000.0
    feed_total_kg_per_fish = (np.array(Feed_arr, dtype=float) / 1000.0) / N_fish
    
    feed_ratio = feed_total_kg_per_fish / W_est[:196]

    ind = [(float(feed_ratio[d]), float(T_arr[d]), float(DO_const), float(Q_const)) for d in range(196)]

    try:
        weights, _, _, _, _, _, _, _ = bio_model.run_sim(
            ind, initial_weight, params.TAN0
        )

        weights = np.array(weights, dtype=float)

        pred_full = weights * 1000.0   # kg -> grams, all 197 points (days 0-196)

        if np.any(~np.isfinite(pred_full)) or np.any(pred_full <= 0):
            return 10.0 

        pred_at_data_days = np.array([pred_full[d] for d in Data_days])
        real_data = np.array(Data)

        # Relative RMSE 
        err = np.sqrt(np.mean(((pred_at_data_days - real_data) / real_data) ** 2))
        
        return err

    except Exception as e:
        return 10.0
    
    

# def tank_error(Feed_arr, T_arr, Data_days, Data, N_fish, DO_const, Q_const):
#     """
#     Compute calibration error for one tank using the full 196-day curve.
#     """
#     initial_weight = Data[0] / 1000.0   # kg
#     days_full = np.arange(196)

#     # Full measured curve from sparse measurements
#     meas_full = np.interp(days_full, Data_days, Data)   # grams

#     # Build feed ratio using same estimated daily weights
#     W_est = meas_full / 1000.0   # kg

#     feed_total_kg_per_fish = (np.array(Feed_arr, dtype=float) / 1000.0) / N_fish
#     feed_ratio = feed_total_kg_per_fish / W_est

#     ind = [(float(feed_ratio[d]), float(T_arr[d]), float(DO_const), float(Q_const)) for d in range(196)]

#     try:
#         weights, feeds_kg, tan_lst, no3_lst, co2_lst, temps, DOs, Q_waters = bio_model.run_sim(
#             ind, initial_weight, params.TAN0
#         )

#         weights = np.array(weights, dtype=float)

#         # validity checks
#         if len(weights) < 196:
#             return 1e12

#         pred_full = weights[:196] * 1000.0   # kg -> grams

#         if np.any(~np.isfinite(pred_full)) or np.any(pred_full <= 0):
#             return 1e12

#         # Full-curve relative RMSE
#         err = np.sqrt(np.mean(((pred_full - meas_full) / meas_full) ** 2))
#         return err

#     except Exception:
#         return 1e12


def fit_evaluation(population):
    """
    Fitness = negative error. Includes equal tank weighting and L2 Regularization.
    """
    fitness_lst = []

    DO_const = 6
    Q_const = 0.6
    T_arr = [24] * 70 + [27] * 126
    Data_days_A = [0, 2, 22, 29, 36, 56, 63, 78, 84, 91, 105, 112, 120, 155, 168, 177, 190, 196]

    # Tank A
    N_fish_A = 66
    Feed_arr_A = [70]*23 + [115]*55 + [129]*6 + [117]*7 + [187]*14 + [124]*7 + [131]*8 + [168]*35 + [208]*13 + [160]*9 + [161]*13 + [168]*6
    Data_A = [14, 14.2, 16, 17.35, 17.9, 20.84, 20.68, 28.8, 25.8, 26, 30, 32, 33.2, 42, 39.7, 40.4, 43, 41.7]

    # Tank B 
    N_fish_B = 40
    Feed_arr_B = [100]*23 + [185]*55 + [201]*6 + [187]*7 + [52]*14 + [57.16]*7 + [53]*8 + [62]*35 + [63]*13 + [68]*9 + [70]*13 + [70]*6
    Data_days_B = [0, 2, 22, 29, 36, 56, 63, 78, 84, 91, 105, 112, 120, 155, 168, 177, 190, 196]
    Data_B =      [49.8, 61, 61.8, 64.4, 72.1, 75.85, 75.4, 84.1, 78.2, 80, 88.5, 78.5, 86, 89.5, 93.25, 97, 96.9, 98.9]

    for cand in population:
        for name, value in zip(PARAM_NAMES, cand):
            setattr(bio_model, name, value)

        try:
            tank_A_err = tank_error(Feed_arr_A, T_arr, Data_days_A, Data_A, N_fish_A, DO_const, Q_const)
            tank_B_err = tank_error(Feed_arr_B, T_arr, Data_days_B, Data_B, N_fish_B, DO_const, Q_const)
            
            base_err = 0.5 * tank_A_err + 0.5 * tank_B_err

            # =======================================================
            # FIX 4: L2 Regularization
            # Penalizes the GA for selecting boundary values
            # =======================================================
            reg_penalty = 0.0
            for name, value in zip(PARAM_NAMES, cand):
                low, high = PARAM_RANGES[name]
                mid = (high + low) / 2.0
                norm_diff = (value - mid) / (high - low)
                reg_penalty += (norm_diff ** 2) * 0.01  # reduced from 0.05 

            total_err = base_err + reg_penalty
            fitness_lst.append(-total_err)

        except Exception:
            fitness_lst.append(-1e12)

    return fitness_lst


# def fit_evaluation(population):
#     """
#     Fitness = negative error between simulated and measured weights.
#     Higher fitness is better.
#     """
#     fitness_lst = []

#     # ===== Real calibration data =====
#     DO_const = 6
#     Q_const = 0.6

#     T_arr = [24] * 70 + [27] * 126
#     Data_days = [0, 2, 22, 29, 36, 56, 63, 78, 84, 91, 105, 112, 120, 155, 168, 177, 190, 196]

#     # ======== Tank A =================

#     N_fish_A = 66
#     Feed_arr_A = [70]*23 + [115]*55 + [129]*6 + [117]*7 + [187]*14 + [124]*7 + [131]*8 + [168]*35 + [208]*13 + [160]*9 + [161]*13 + [168]*6
#     Data_A = [14, 14.2, 16, 17.35, 17.9, 20.84, 20.68, 28.8, 25.8, 26, 30, 32, 33.2, 42, 39.7, 40.4, 43, 41.7]

#     # ======== Tank B =================

#     N_fish_B = 40
#     Feed_arr_B = [100]*23 + [185]*55 + [201]*6 + [187]*7 + [52]*14 + [57.16]*7 + [53]*8 + [62]*35 + [63]*13 + [68]*9 + [70]*13 + [70]*6
#     Data_B = [49.8, 61, 61.8, 64.4, 72.1, 75.85, 75.4, 84.1, 78.2, 80, 88.5, 78.5, 86, 89.5, 93.25, 97, 96.9, 98.9]

#     tanks = [
#     (Feed_arr_A, Data_A, N_fish_A),
#     (Feed_arr_B, Data_B, N_fish_B),
#     ]

#     for cand in population:
#         for name, value in zip(PARAM_NAMES, cand):
#             # This untangles each param
#             setattr(bio_model, name, value)

#         try:
#             total_err = 0.0

#             total_err = 0.1 * tank_error(
#                 Feed_arr_A, T_arr, Data_days, Data_A, N_fish_A, DO_const, Q_const
#             ) + 0.9 * tank_error(
#                 Feed_arr_B, T_arr, Data_days, Data_B, N_fish_B, DO_const, Q_const
#             )

#             fitness_lst.append(-total_err)

#         except Exception:
#             fitness_lst.append(-1e12)

#     return fitness_lst


def select_parent(population, fitness, k=3):
    """
    Tournament selection: choose k random individuals and return
    a copy of the one with the highest fitness.
    """
    candidates = rand.sample(range(len(population)), k)
    best_index = max(candidates, key=lambda i: fitness[i])
    return population[best_index]


def crossover(p1, p2, crossover_rate: float = 0.9):
    """
    Blend crossover between two parent parameter tuples.
    """
    if rand.random() > crossover_rate:
        return p1[:], p2[:]

    blend = rand.random()
    c1 = []
    c2 = []

    for v1, v2 in zip(p1, p2):

        c1.append(blend*v1 + (1-blend)*v2)
        c2.append(blend*v2 + (1-blend)*v1)

    return c1, c2


def mutate(ind):
    """
    Mutation with parameter-specific std and rate scales.
    Base std/rate come from percent of range, then each parameter
    gets multiplied by its own scale factor.
    """
    new_ind = []

    for name, value in zip(PARAM_NAMES, ind):
        low, high = PARAM_RANGES[name]

        std = PARAM_STD_SCALE.get(name, 1.0)
        rate = PARAM_RATE_SCALE.get(name, 1.0)

        if rand.random() < rate:
            value += rand.gauss(0, std)

        value = min(max(value, low), high)
        new_ind.append(value)

    return new_ind


def run_ga(pop_size, num_gens, seed_ind=None):
    """
    Run calibration GA.
    Returns best parameter vector and best error.
    """
    population = generate_population(pop_size, seed_ind)

    best_ind = None
    best_fit = -np.inf

    for gen in range(num_gens):
        fits = fit_evaluation(population)

        idx_best = int(np.argmax(fits))
        if fits[idx_best] > best_fit:
            best_fit = fits[idx_best]
            best_ind = population[idx_best][:]

        new_pop = [best_ind[:]]

        while len(new_pop) < pop_size:
            p1 = select_parent(population, fits)
            p2 = select_parent(population, fits)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

        summary = ", ".join(
            f"{name}={value:.4f}" for name, value in zip(PARAM_NAMES, best_ind)
        )
        print(f"Gen {gen+1}: best_error={-best_fit:.6f}, {summary}")

    return best_ind, -best_fit