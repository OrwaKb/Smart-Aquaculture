
from params import *
import math

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


def TAN_to_UIA(TAN: float) -> float:
    """
    Compute UIA (unionized ammonia) from TAN using pH and pKa:
        UIA = TAN / (1 + 10^(pKa - pH))
    Based on M. Saadi's code.
    """
    UIA = TAN / (1 + 10**(PK_a - PH))
    return max(UIA, 0.0)
    #return TAN * frac_UIA


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

    ####################### COSTS CALCULATION ########################
    # Taken from m.saadi's code
    
    #electricity cost from heating 
    #heat_flux = area * 12.12 * (T_env - T)
    #heat_cost = max((heat_flux/1000) * electricity_price * 24 * days,0)
    
    #electricity price from adding Oxygen
    #DO_cost = ((DO / 1000 /24)/ 0.46) * electricity_price  * 24 * days

    #electricity = heat_cost + DO_cost
    #=================================================================

    #total_cost = feed_cost + electricity

    temp_cost = cT * max(0.0, T - T_env) * days
    do_cost   = cDO * max(0.0, DO - DO_base) ** 2 * days

    total_cost = feed_cost + temp_cost + do_cost

    return revenue - total_cost
#========================================================================================