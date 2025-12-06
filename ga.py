
from mpc import MPC
from params import *
import numpy as np
import random as rand

# =========================================================================
# Genetic Algorithm – temperature & DO optimization
# =========================================================================


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


def fit_evaluation(population,
                   total_days_GA,
                   pred_horiz_GA,
                   feeding_list_GA,
                   w0,
                   TAN0):
    """
    Given a population, return a list of fitness values.
    In this case the fitness of a (T, DO) pair is its resulting profit
    when used inside the MPC controller.
    """
    fitness_lst = []
    for pair in population:
        weights, feeds, profit_lst, tan_lst = MPC(
            total_days_GA, pred_horiz_GA, feeding_list_GA, pair[0], pair[1], w0, TAN0
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


def run_ga(population_size, num_generations, Temp_lst, DO_lst, 
           total_days_GA, pred_horiz_GA, feeding_list_GA, w0, TAN0):

    population = generate_population(population_size, Temp_lst, DO_lst)

    best_ind = None
    best_fit = -np.inf

    for gen in range(num_generations):
        fits = fit_evaluation(population, total_days_GA,
            pred_horiz_GA,
            feeding_list_GA,
            w0,
            TAN0
        )

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
        print(f"Gen {gen}: best_fit={best_fit:.3f}, T={best_ind[0]:.2f}, DO={best_ind[1]:.2f}")

    return best_ind, best_fit

#==========================================================================================================