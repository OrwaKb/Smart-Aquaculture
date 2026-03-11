import params
import random as rand
import numpy as np
import bio_model

# =========================================================================
# Genetic Algorithm – Calibration Our Model for Min Error
# =========================================================================

PARAM_NAMES = [
    "m", "n", "h", "j",
    "k_min", "k", 
    "T_opt", "T_min","T_max"
]

PARAM_RANGES = {
    "m": (0.1, 4),
    "n": (0.1, 4),
    "h": (0.1, 2),
    "j": (0.001, 0.05),
    "k_min": (0.0001, 0.005),
    "k": (2, 8),
    "T_opt" : (10,30),
    "T_min" : (10,30),
    "T_max" : (30, 50),

}

PARAM_STD_SCALE = {
    "m": 0.05,
    "n": 0.05,
    "h": 0.03,
    "k_min": 0.0001,
    "j": 0.001,
    "k": 0.10,
    "T_opt": 1,
    "T_min": 1,
    "T_max":1
}

PARAM_RATE_SCALE = {
    "m": 0.2,
    "n": 0.2,
    "h": 0.02,
    "k_min": 0.02,
    "j": 0.2,
    "k": 0.2,
    "T_opt": 0.2,
    "T_min": 0.2,
    "T_max": 0.2
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
    Compute calibration error for one tank.
    """
    initial_weight = Data[0] / 1000.0   # kg
    days_full = np.arange(196)

    # Interpolate measured weights across all days
    weights_measured_interp = np.interp(days_full, Data_days, Data)  # grams

    # Rebuild stepped weight estimate
    weights_measured_step = []
    i = 0
    for idx in range(len(Data_days) - 1):
        gap = Data_days[idx + 1] - Data_days[idx]
        for _ in range(gap):
            weights_measured_step.append(Data[i])
        i += 1

    W_est = np.array(weights_measured_step[:196], dtype=float) / 1000.0  # kg

    feed_total_kg_per_fish = (np.array(Feed_arr, dtype=float) / 1000.0) / N_fish
    feed_ratio = feed_total_kg_per_fish / W_est

    ind = [(float(feed_ratio[d]), float(T_arr[d]), float(DO_const), float(Q_const)) for d in range(196)]

    weights, feeds_kg, tan_lst, no3_lst, co2_lst, temps, DOs, Q_waters = bio_model.run_sim(
        ind, initial_weight, params.TAN0
    )

    weights = np.array(weights, dtype=float)

    # Basic validity checks
    if len(weights) <= max(Data_days):
        return 1e12

    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        return 1e12

    # Convert simulated kg -> grams for comparison
    pred = weights[Data_days] * 1000.0
    meas = np.array(Data, dtype=float)

    if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
        return 1e12

    # Relative RMSE
    err = np.sqrt(np.mean(((pred - meas) / meas) ** 2))
    return err


def fit_evaluation(population):
    """
    Fitness = negative error between simulated and measured weights.
    Higher fitness is better.
    """
    fitness_lst = []

    # ===== Real calibration data =====
    DO_const = 6
    Q_const = 0.6

    T_arr = [24] * 70 + [27] * 126
    Data_days = [0, 2, 22, 29, 36, 56, 63, 78, 84, 91, 105, 112, 120, 155, 168, 177, 190, 196]

    # ======== Tank A =================

    N_fish_A = 66
    Feed_arr_A = [70]*23 + [115]*55 + [129]*6 + [117]*7 + [187]*14 + [124]*7 + [131]*8 + [168]*35 + [208]*13 + [160]*9 + [161]*13 + [168]*6
    Data_A = [14, 14.2, 16, 17.35, 17.9, 20.84, 20.68, 28.8, 25.8, 26, 30, 32, 33.2, 42, 39.7, 40.4, 43, 41.7]

    # ======== Tank B =================

    N_fish_B = 40
    Feed_arr_B = [100]*23 + [185]*55 + [201]*6 + [187]*7 + [52]*14 + [57.16]*7 + [53]*8 + [62]*35 + [63]*13 + [68]*9 + [70]*13 + [70]*6
    Data_B = [49.8, 61, 61.8, 64.4, 72.1, 75.85, 75.4, 84.1, 78.2, 80, 88.5, 78.5, 86, 89.5, 93.25, 97, 96.9, 98.9]

    tanks = [
    (Feed_arr_A, Data_A, N_fish_A),
    (Feed_arr_B, Data_B, N_fish_B),
    ]

    for cand in population:
        for name, value in zip(PARAM_NAMES, cand):
            # This untangles each param
            setattr(bio_model, name, value)

        try:
            total_err = 0.0

            for Feed_arr, Data, N_fish in tanks:
                total_err += tank_error(
                    Feed_arr, T_arr, Data_days, Data, N_fish, DO_const, Q_const
                )

            fitness_lst.append(-total_err)

        except Exception:
            fitness_lst.append(-1e12)

    return fitness_lst


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