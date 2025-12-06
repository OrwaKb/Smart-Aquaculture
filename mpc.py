
from bio_model import dwdt, TAN_to_UIA, update_TAN, profit
from params import *
from scipy.integrate import solve_ivp
import numpy as np

# =========================================================================
# MPC Function
# =========================================================================


def MPC(total_days, pred_horiz, feeding_list, T, DO, initial_weight, initial_Tan, stop = True):
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
        if stop == True:
            if daily_profit < 0:
                break

    return weights, feeds, profit_lst, tan_lst

#==============================================================================================