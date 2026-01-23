
from params import *
import numpy as np
import random as rand
from bio_model import run_sim, profit, TAN_to_UIA

# =========================================================================
# Genetic Algorithm – temperature & DO optimization
# =========================================================================


def generate_genome(first_lst: list, second_lst: list, third_lst: list, fourth_lst:list):
    """
    Generate a candidate solution (feed,T,DO,Q) by random choice
    from the provided discrete lists.
    """
    return rand.choice(first_lst), rand.choice(second_lst), rand.choice(third_lst), rand.choice(fourth_lst)


def generate_population(size: int, horizon: int, first_lst: list, second_lst: list, third_lst: list, fourth_lst: list, seed_ind=None):
    """ Generate a population of candidate solutions using the predefined generate_genome function.
    * 7 because we want to solutins for the prediction horizen all at a time. change number to word later """
    population = []

    # If we have a seed (previous best solution), add it to the population
    if seed_ind is not None:
        population.append(seed_ind)

    # Fill the rest with random individuals
    while len(population) < size: 
        population.append([generate_genome(first_lst, second_lst, third_lst, fourth_lst) for _ in range(horizon)]) 
     
    return population


def fit_evaluation(population, initial_weight, initial_tan):
    '''
    fit evaluation generates a population with 3 charactaristics, feed, Temp and DO and gives us the 
    fitness
    
    :param size: describes how big we want the generation to be 
    :param horizon: the horizon, meaning how many lists we want, for how many days
    we want this to be.
    '''
    fitness_lst = []
    
    for ind in population:
        weights, feeds, tan_lst, no3_lst, co2_lst, Ts, DOs, Qs = run_sim(ind, initial_weight, initial_tan)
        profit_ind = profit(weights[-1], weights[0], feeds, Ts, DOs, Qs,len(ind))

        # ================ Nitrate Penalty ====================================

        NO3_soft = 0.7 * NO3_crit
        lam_no3 = 1e6  # tune

        for n in no3_lst:
            if n > NO3_soft:
                profit_ind -= lam_no3 * (n - NO3_soft)**2


        # ================ CO2 Penalty ========================================

        CO2_soft = 0.7 * CO2_crit
        lam_co2 = 1e1  # for tuning

        for c in co2_lst:
            if c > CO2_soft:
                profit_ind -= lam_co2 * (c - CO2_soft)**2

        # ================ UIA Penalty ========================================

        #uia_lst = [TAN_to_UIA(t) for t in tan_lst]
        
        #uia_soft = 0.7 * UIA_crit
        #lam_uia  = 5e2  # for tuning

        #for u in uia_lst:
            #if u > UIA_max:
                #profit_ind -= 1e8

            #elif u > uia_soft:
                #profit_ind -= lam_uia * (u - uia_soft)**2

        # ================ TAN Penalty ========================================

        #TAN_soft = 3.0     
        #TAN_hard = 6.0    
        #lam_tan  = 1e1     

        #for t in tan_lst:
            #if t > TAN_hard:
                #profit_ind -= 1e4
            #elif t > TAN_soft:
                #profit_ind -= lam_tan * (t - TAN_soft)**2

        # =====================================================================

        fitness_lst.append(profit_ind)

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

    c1, c2 = [], []
    for (F1, T1, DO1, Q1), (F2, T2, DO2, Q2) in zip(p1, p2):
        blend = rand.random()
        F = blend*F1 + (1-blend)*F2
        T = blend*T1 + (1-blend)*T2
        DO = blend*DO1 + (1-blend)*DO2
        Q = blend*Q1 + (1-blend)*Q2

        F = min(max(F, feed_min), feed_max)
        T = min(max(T, T_min), T_max)
        DO = min(max(DO, DO_min), DO_max)
        Q = min(max(Q, Q_water_min), Q_water_max)

        c1.append((F, T, DO, Q))

        F = blend*F2 + (1-blend)*F1
        T = blend*T2 + (1-blend)*T1
        DO = blend*DO2 + (1-blend)*DO1
        Q = blend*Q2 + (1-blend)*Q1

        F = min(max(F, feed_min), feed_max)
        T = min(max(T, T_min), T_max)
        DO = min(max(DO, DO_min), DO_max)
        Q = min(max(Q, Q_water_min), Q_water_max)

        c2.append((F, T, DO, Q))
        
    return c1, c2


def mutate(ind,
           mut_rate_T: float = 0.2,
           mut_rate_DO: float = 0.2,
           mut_rate_f: float = 0.2,
           mut_rate_Q: float = 0.2,
           std_T: float = 0.1,
           std_DO: float = 0.1,
           std_f: float = 0.01,
           std_Q: float = 0.05):
    """
    Mutation operator for a HORIZON individual.
    ind is a list of daily tuples: [(feed, T, DO), (feed, T, DO), ...]
    Applies small Gaussian noise to each day's variables with given probabilities.
    """

    new_ind = []

    for day_gene in ind:
        feed, T, DO, Q = day_gene

        if rand.random() < mut_rate_T:
            T += rand.gauss(0, std_T)
        if rand.random() < mut_rate_DO:
            DO += rand.gauss(0, std_DO)
        if rand.random() < mut_rate_f:
            feed += rand.gauss(0, std_f)
        if rand.random() < mut_rate_Q:
            Q += rand.gauss(0, std_Q)

        # clip to bounds
        T = min(max(T, T_min), T_max)
        DO = min(max(DO, DO_min), DO_max)
        feed = min(max(feed, feed_min), feed_max)
        Q = min(max(Q, Q_water_min), Q_water_max)

        new_ind.append((feed, T, DO, Q))

    return new_ind


def run_ga(pop_size, num_gens, horizon, feed_lst, temp_lst, do_lst, q_water_lst, w0, TAN0, seed_ind=None):

    population = generate_population(pop_size, horizon, feed_lst, temp_lst, do_lst, q_water_lst, seed_ind)

    best_ind = None
    best_fit = -np.inf

    for gen in range(num_gens):
        fits = fit_evaluation(population, w0, TAN0)

        idx_best = int(np.argmax(fits))
        if fits[idx_best] > best_fit:
            best_fit = fits[idx_best]
            best_ind = population[idx_best]

        new_pop = [best_ind]  # elitism

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

        # print day-0 decision 
        F0, T0, DO0, Q0 = best_ind[0]
        #print(f"Gen {gen}: best_fit={best_fit:.4f}, day0: F={F0:.4f}, T={T0:.2f}, DO={DO0:.2f}")

    return best_ind, best_fit

#==========================================================================================================