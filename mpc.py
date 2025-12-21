
from bio_model import dwdt, TAN_to_UIA, update_water_quality, profit
from ga import run_ga
from params import *
from scipy.integrate import solve_ivp

# =========================================================================
# MPC Function
# =========================================================================


def MPC(total_days, pred_horiz, feeding_lst, T_lst, DO_lst,
        initial_weight, initial_Tan,
        population_size, num_gens,
        stop=False):

    weights      = [float(initial_weight)]
    tan_lst      = [float(initial_Tan)]
    no3_lst      = [float(NO3_0)]
    feeds_kg     = []
    feed_prec    = []
    applied_plan = []   # store applied (F,T,DO) each day
    profit_cum   = []
    t_span       = (0.0, 1.0)

    prev_best_ind = None # Initialize variable to store previous plan

    for day in range(total_days):

        w_current   = float(weights[-1])
        TAN_current = float(tan_lst[-1])

        seed_ind = None
        if prev_best_ind is not None:
            # Shift the plan: Drop Day 0, keep Day 1-6
            seed_ind = prev_best_ind[1:] 
            # We need to add a new guess for the new "last day" of the horizon
            # We can just duplicate the last known good move
            seed_ind.append(prev_best_ind[-1])

        # Run GA
        best_ind, best_fit = run_ga(
            population_size, num_gens, pred_horiz,
            feeding_lst, T_lst, DO_lst,
            w_current, TAN_current, 
            seed_ind=seed_ind
        )

        # Apply only day-0 action
        F0, T0, DO0 = map(float, best_ind[0])
        applied_plan.append((F0, T0, DO0))

        UIA_now = TAN_to_UIA(TAN_current)
        if UIA_now > UIA_crit:
            break

        sol = solve_ivp(
            lambda t, w: dwdt(t, w, UIA_now, F0, T0, DO0),
            t_span,
            [w_current],
            t_eval=[t_span[1]]
        )
        if not sol.success:
            break

        w_end = float(sol.y[0, -1])

        # Save this result for the next loop iteration
        prev_best_ind = best_ind

        # feed in kg/day 
        feed_kg = float(F0 * w_current)

        TAN_next, NO3_next = update_water_quality(TAN_current, no3_lst[-1], feed_kg) #float(update_TAN(TAN_current, feed_kg))

        # daily + cumulative profit
        daily_profit = float(profit(w_end, w_current, [feed_kg], [T0], [DO0], 1))
        cumulative_prof = (profit_cum[-1] if profit_cum else 0.0) + daily_profit

        # check constraint
        if TAN_to_UIA(TAN_next) > UIA_crit:
            break

        if stop and daily_profit < 0:
            break

        if NO3_next > NO3_crit:
             # Later
             pass
        
        # commit day
        weights.append(w_end)
        tan_lst.append(TAN_next)
        no3_lst.append(NO3_next)
        feeds_kg.append(feed_kg)
        feed_prec.append(F0)
        profit_cum.append(cumulative_prof)
        
        print(f"Day {day}: F={F0:.4f}, T={T0:.2f}, DO={DO0:.2f}, w={w_end:.4f}, profit={daily_profit:.4f}")

    return weights, feeds_kg, applied_plan, profit_cum, tan_lst, no3_lst, feed_prec

        
#==============================================================================================