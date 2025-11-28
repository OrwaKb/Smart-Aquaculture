"""
Model Predictive System (MPC)

@author: Orwa Kblawe
Date: Nov 2025
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================================
# Parameters and constraints
# =========================================================================

h       = 0.8
p       = 1.5
b       = 0.62
a       = 0.53
m       = 0.67
n       = 0.81

T       = 30
T_min   = 24
T_opt   = 33
T_max   = 40

k_min   = 0.00133
j       = 0.0132

DO      = 1.5
DO_crt  = 0.3
DO_min  = 1.0

UIA_crit = 0.06
UIA_max  = 1.4

p_feed   = 0.7   # feed price per kg
p_fish   = 2.0   # fish price per kg

w0       = 0.0001  # initial weight in kg

# TAN / biofilter parameters
Nf       = 0.03    # [g TAN / g feed]
V_water  = 100.0   # [m^3] system water volume
V_BF     = 10.0    # [m^3] biofilter volume
n_BF     = 0.5     # [g TAN / (m^3 BF · day)]
k_BF     = 0.7     # exponent
TAN0     = 0.1     # initial TAN

# =========================================================================
# Equations Used - Define
# =========================================================================

def tao(t: float) -> float:
    """Temperature effect tao(T)."""
    k = 4.6
    if t > T_opt:
        return math.exp(-k * ((t - T_opt) / (T_max - T_opt))**4)
    if t < T_opt:
        return math.exp(-k * ((T_opt - t) / (T_opt - T_min))**4)
    return 1.0


def segma(x: float) -> float:
    """Dissolved oxygen function σ(DO)."""
    if x > DO_crt:
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


def dwdt(t: float, w: float, UIA: float, f: float) -> float:
    """
    Main growth ODE: dw/dt as a function of weight, UIA and feed ratio.
    f is feeding ratio (fraction of body weight per day).
    """
    growth = h * p * f * b * (1 - a) * tao(T) * segma(DO) * v(UIA) * w**m
    loss   = k_min * math.exp(j * (T - T_min)) * w**n
    return growth - loss


def profit(w_final: float, w_initial: float, f_list) -> float:
    """
    Profit for a period:
    w_final, w_initial in kg; f_list is a list of feed amounts in kg.
    """
    total_feed = sum(f_list)
    return (w_final - w_initial) * p_fish - total_feed * p_feed

# =========================================================================
# Main Function
# =========================================================================

t_span       = [0, 1]                 # integrate over 1 day
total_days   = 200
pred_horiz   = 7
feeding_list = np.arange(0.01, 0.2, 0.005)

weights   = [w0]
tan_lst   = [TAN0]
feeds     = []
profit_lst = []

for day in range(total_days):
    """
    For each day, we run an MPC step:
    - Predict over a horizon for each candidate feed ratio.
    - Choose the ratio that maximizes predicted profit (with UIA constraint).
    - Apply that feed to the real system for one day.
    """
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
                lambda t, w: dwdt(t, w, UIA_current, feed),
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
                profit(pred_weights[-1], pred_weights[0], pred_feeds_kg)
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
        lambda t, w: dwdt(t, w, UIA_real, max_feed),
        t_span,
        [w_current]
    )
    w_next = sol_real.y[0, -1]

    # Update daily parameters
    feed_today_kg = max_feed * w_current
    TAN_current   = update_TAN(TAN_current, feed_today_kg)
    daily_profit  = profit(w_next, w_current, [feed_today_kg])

    weights.append(w_next)
    tan_lst.append(TAN_current)

    if profit_lst:
        profit_lst.append(profit_lst[-1] + daily_profit)
    else:
        profit_lst.append(daily_profit)

    # Optional: stop if profit becomes negative
    # if daily_profit < 0:
    #     break

# =========================================================================
# Plots
# =========================================================================

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

# =========================================================================
# End Plots
# =========================================================================
