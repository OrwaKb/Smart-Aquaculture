"""
Model Predictive Control (MPC) with Genetic Algorithm (GA)
for optimal temperature and dissolved oxygen selection.

@author: Orwa Kblawe
Date: Nov 2025
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand
from scipy.integrate import solve_ivp

# =========================================================================
# Parameters and constraints
# =========================================================================

h        = 0.8
p        = 1.5
b        = 0.62
a        = 0.53
m        = 0.67
n        = 0.81

T_min    = 24
T_opt    = 33   # intrinsic optimal temperature used in tao(T)
T_max    = 40
T_env    = 28   # Temperature of the room

k_min    = 0.00133
j        = 0.0132

DO_crt   = 3.0
DO_min   = 1.0
DO_max   = 5.0
DO_base  = 2.0   # Baseline DO
 
cT       = 0.00  # cost per celcious * day                    #
cDO      = 0.06  # cost per mg/l                              #

UIA_crit = 0.06
UIA_max  = 1.4

p_feed   = 0.7  # feed price per kg
p_fish   = 2.0  # fish price per kg

# TAN / biofilter parameters
Nf       = 0.03   # [g TAN / g feed]
V_water  = 100.0  # [m^3] system water volume
V_BF     = 10.0   # [m^3] biofilter volume
n_BF     = 0.5    # [g TAN / (m^3 BF · day)]
k_BF     = 0.7    # exponent

# =========================================================================
# Biological / MPC equations
# =========================================================================

def tao(t: float) -> float:
    """Temperature effect τ(T)."""
    k = 4.6
    if t > T_opt:
        return math.exp(-k * ((t - T_opt) / (T_max - T_opt))**4)
    if t < T_opt:
        return math.exp(-k * ((T_opt - t) / (T_opt - T_min))**4)
    return 1.0


def segma(x: float) -> float:
    """Dissolved oxygen limitation function σ(DO)."""
    if x >= DO_crt:
        return 1.0
    if DO_min < x < DO_crt:
        return (x - DO_min) / (DO_crt - DO_min)
    return 0.0


def v(x: float) -> float:
    """Unionized ammonia effect v(UIA)."""
    if x < UIA_crit:
        return 1.0
    if UIA_crit < x < UIA_max:
        return (UIA_max - x) / (UIA_max - UIA_crit)
    return 0.0


def BF_capacity(TAN_prev: float) -> float:
    """Biofilter capacity BF_{t-1}."""
    return n_BF * (TAN_prev ** k_BF)


def update_TAN(TAN_prev: float, feed_kg: float) -> float:
    """Update TAN based on previous TAN and feed amount (kg)."""
    feed_g = feed_kg * 1000.0  # convert to grams

    input_term   = (Nf * feed_g) / V_water
    BF_prev      = BF_capacity(TAN_prev)
    removal_term = (BF_prev * V_BF) / V_water

    TAN_next = TAN_prev + input_term - removal_term
    return max(TAN_next, 0.0)


def TAN_to_UIA(TAN: float, frac_UIA: float = 0.05) -> float:
    """Approximate UIA as a fixed fraction of TAN."""
    return TAN * frac_UIA


def dwdt(t: float, w: float, UIA: float, f: float, T: float, DO: float) -> float:
    """
    Main growth ODE: dw/dt as a function of weight, UIA and feed ratio.
    f is feeding ratio (fraction of body weight per day).
    """
    growth = h * p * f * b * (1 - a) * tao(T) * segma(DO) * v(UIA) * w**m
    loss   = k_min * math.exp(j * (T - T_min)) * w**n
    return growth - loss


def profit(w_final: float,
           w_initial: float,
           f_list,
           T: float,
           DO: float,
           days: int) -> float:
    """
    Bio-economic profit for a given period.

    :param w_final:  final fish weight (kg)
    :param w_initial: initial fish weight (kg)
    :param f_list:   list of feed amounts in kg over the period
    :param T:        water temperature (°C)
    :param DO:       dissolved oxygen (mg/L)
    :param days:     number of days in this profit period
    """
    total_feed = sum(f_list)

    revenue = (w_final - w_initial) * p_fish
    feed_cost = total_feed * p_feed
    temp_cost = cT * max(0.0, T - T_env) * days
    do_cost   = cDO * max(0.0, DO - DO_base)**2 * days

    total_cost = feed_cost + temp_cost + do_cost

    return revenue - total_cost

# =========================================================================
# Genetic Algorithm – temperature & DO optimization
# =========================================================================

Genome     = []  # list of possible solutions (not used explicitly)
population = []  # list of genomes


def generate_genome(first_lst: list, second_lst: list):
    """
    Generate a candidate solution (T, DO) by random choice
    from the provided discrete lists.
    """
    return rand.choice(first_lst), rand.choice(second_lst)


def generate_population(size: int, first_lst: list, second_lst: list):
    """
    Generate a population of candidate solutions using the predefined
    generate_genome function.
    """
    return [generate_genome(first_lst, second_lst) for _ in range(size)]


def fit_evaluation(population):
    """
    Given a population, return a list of fitness values.
    In this case the fitness of a (T, DO) pair is its resulting profit
    when used inside the MPC controller.
    """
    fitness_lst = []
    for pair in population:
        weights, feeds, profit_lst, tan_lst = MPC(
            total_days, pred_horiz, feeding_list, pair[0], pair[1], w0, TAN0
        )
        fitness = profit_lst[-1]
        fitness_lst.append(fitness)
    return fitness_lst


def select_parent(population, fitness, k=3):
    """
    Tournament selection: choose k random individuals and return
    a copy of the one with the highest fitness.
    """
    candidates = rand.sample(range(len(population)), k)
    best_index = max(candidates, key=lambda i: fitness[i])
    return population[best_index][:]


def crossover(p1, p2, crossover_rate: float = 0.9):
    """
    Blend crossover between two parents to create new offspring.

    With probability (1 - crossover_rate), parents are copied directly.
    Otherwise, a random blend factor is used to linearly combine T and DO.
    """
    if rand.random() > crossover_rate:
        return p1[:], p2[:]  # copies

    blend_fac = rand.random()

    # create two children by mixing p1 and p2
    c1 = [
        blend_fac * p1[0] + (1 - blend_fac) * p2[0],  # T
        blend_fac * p1[1] + (1 - blend_fac) * p2[1]   # DO
    ]

    c2 = [
        blend_fac * p2[0] + (1 - blend_fac) * p1[0],
        blend_fac * p2[1] + (1 - blend_fac) * p1[1]
    ]

    return c1, c2


def mutate(ind, mut_rate_T: float = 0.2, mut_rate_DO: float = 0.2,
           std_T: float = 0.5, std_DO: float = 0.2):
    """
    Mutation operator: introduces small random changes to offspring.
    This helps avoid local optima and maintains diversity.

    :param ind:        individual [T, DO] that we want to mutate
    :param mut_rate_T: probability of mutating the temperature T
    :param mut_rate_DO: probability of mutating the dissolved oxygen DO
    :param std_T:      standard deviation of Gaussian noise added to T
    :param std_DO:     standard deviation of Gaussian noise added to DO
    """
    T, DO = ind
    if rand.random() < mut_rate_T:
        T += rand.gauss(0, std_T)
    if rand.random() < mut_rate_DO:
        DO += rand.gauss(0, std_DO)

    # clip to bounds
    T = min(max(T, T_min), T_max)
    DO = min(max(DO, DO_min), DO_max)

    return [T, DO]

# =========================================================================
# MPC Function
# =========================================================================


def MPC(total_days, pred_horiz, feeding_list, T, DO, initial_weight, initial_Tan):
    """
    Model Predictive Control (MPC) system.

    Evaluates and predicts what will happen in the next pred_horiz days
    for each candidate feed ratio and chooses the best feed for each day.

    This function is also used by the GA as the fitness evaluator,
    returning the cumulative profit.

    :param total_days:     total days of the simulation
    :param pred_horiz:     prediction horizon (days ahead to simulate)
    :param feeding_list:   list of feed ratios tested
    :param T:              water temperature
    :param DO:             dissolved oxygen
    :param initial_weight: initial weight of the fish (kg)
    :param initial_Tan:    initial TAN of the tank
    """
    weights    = [initial_weight]
    tan_lst    = [initial_Tan]
    feeds      = []
    profit_lst = []
    t_span     = [0, 1]  # integrate over 1 day

    for day in range(total_days):
        # For each day, run an MPC step:
        # - Predict over a horizon for each candidate feed ratio.
        # - Choose the ratio that maximizes predicted profit (with UIA constraint).
        # - Apply that feed to the real system for one day.
        w_current   = weights[-1]
        TAN_current = tan_lst[-1]
        pred_profit = []

        # --- MPC prediction: try each candidate feed ratio ---
        for feed in feeding_list:
            pred_weights   = [w_current]
            pred_feeds_kg  = []
            pred_tan       = [TAN_current]
            w_tmp          = w_current
            tan_tmp        = TAN_current
            valid          = True

            # Predict over the horizon
            for _ in range(pred_horiz):
                UIA_current = TAN_to_UIA(tan_tmp)

                # UIA constraint
                if UIA_current > UIA_crit:
                    valid = False
                    break

                # Integrate one day ahead
                sol = solve_ivp(
                    lambda t, w: dwdt(t, w, UIA_current, feed, T, DO),
                    t_span,
                    [w_tmp]
                )
                w_tmp = sol.y[0, -1]
                pred_weights.append(w_tmp)

                # Update TAN and feed
                feed_kg = feed * w_tmp
                tan_tmp = update_TAN(tan_tmp, feed_kg)
                pred_tan.append(tan_tmp)
                pred_feeds_kg.append(feed_kg)

            if valid and pred_feeds_kg:
                pred_profit.append(
                    profit(pred_weights[-1], pred_weights[0], pred_feeds_kg, T, DO, pred_horiz)
                )
            else:
                # strongly penalize this feed so MPC will not choose it
                pred_profit.append(-1e9)

        # Choose feed with maximum predicted profit
        max_index = int(np.argmax(pred_profit))
        max_feed  = feeding_list[max_index]
        feeds.append(max_feed)

        # Apply chosen feed to real system for one day
        UIA_real = TAN_to_UIA(TAN_current)
        sol_real = solve_ivp(
            lambda t, w: dwdt(t, w, UIA_real, max_feed, T, DO),
            t_span,
            [w_current]
        )
        w_next = sol_real.y[0, -1]

        # Update daily parameters
        feed_today_kg = max_feed * w_current
        TAN_current   = update_TAN(TAN_current, feed_today_kg)
        daily_profit  = profit(w_next, w_current, [feed_today_kg], T, DO, 1)

        weights.append(w_next)
        tan_lst.append(TAN_current)

        if profit_lst:
            profit_lst.append(profit_lst[-1] + daily_profit)
        else:
            profit_lst.append(daily_profit)

        # Optional: stop if profit becomes negative
        if daily_profit < 0:
            break

    return weights, feeds, profit_lst, tan_lst

# =========================================================================
# Main
# =========================================================================

total_days      = 10
pred_horiz      = 7
feeding_list    = np.arange(0.01, 0.2, 0.01)
population_size = 20
num_generations = 10

DO_lst   = np.arange(DO_min, DO_max, 0.1)
Temp_lst = np.arange(T_min, T_max, 0.5)

TAN0 = 0.1     # initial TAN
w0   = 0.0001  # initial weight in kg

population = generate_population(population_size, Temp_lst, DO_lst)
fit_lst    = fit_evaluation(population)

best_ind = None
best_fit = -np.inf

for gen in range(num_generations):
    fits = fit_evaluation(population)

    # track global best
    idx_best = int(np.argmax(fits))
    if fits[idx_best] > best_fit:
        best_fit = fits[idx_best]
        best_ind = population[idx_best]

    new_pop = []
    # elitism: keep the best
    new_pop.append(best_ind)

    # create rest of population
    while len(new_pop) < population_size:
        p1 = select_parent(population, fits)
        p2 = select_parent(population, fits)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_pop.append(c1)
        if len(new_pop) < population_size:
            new_pop.append(c2)

    population = new_pop

print(np.hstack(best_ind))

# =========================================================================
# Plots (optional – currently commented out)
# =========================================================================
'''
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# 1. Weight Plot
ax1.plot(range(len(weights)), weights, label='Weight', color='blue')
ax1.set_ylabel('Weight (kg)')
ax1.set_title(f'Simulation Results ({total_days} Days)')
ax1.grid(True)
ax1.legend()

# 2. Profit Plot
ax2.plot(range(len(profit_lst)), profit_lst, label='Cumulative Profit', color='green')
ax2.set_ylabel('Profit ($)')
ax2.grid(True)
ax2.legend()

# 3. Feed Plot
ax3.plot(range(len(feeds)), feeds, label='Feed Ratio', color='red')
ax3.set_ylabel('Feed (fraction of BW)')
ax3.grid(True)
ax3.legend()

# 4. TAN / UIA Plot
UIA_list = [TAN_to_UIA(t) for t in tan_lst]

ax4.plot(range(len(tan_lst)), tan_lst, label='TAN (mg/L)', color='purple')
ax4.plot(range(len(UIA_list)), UIA_list, label='UIA (mg/L)', color='orange', linestyle='--')
ax4.set_ylabel('TAN / UIA')
ax4.set_xlabel('Time (days)')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()
'''
# =========================================================================
# End Plots
# =========================================================================
