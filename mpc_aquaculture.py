"""
Model Predictive System (MPC) 

@author: Orwa Kblawe 
Date: Nov 2025
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#=========================================================================
# Parameters and constraints
#=========================================================================

h     = 0.8
p     = 1.5        
b     = 0.62
a     = 0.53
m     = 0.67
n     = 0.81
T     = 30          
T_min = 24
k_min = 0.00133
j     = 0.0132
T_opt = 33
T_max = 40
DO    = 1.5
DO_crt= 0.3
DO_min= 1
UIA_crit=0.06
UIA_max=1.4
p_feed= 0.7                        # feed price per kg
p_fish= 2                          # fish price per kg
w0    = 0.0001                     # initial weight in kg
Nf    = 0.03                       # [g TAN / g feed]  <-- example value
V_water= 100.0                     # [m^3] system water volume
V_BF = 10.0                        # [m^3] biofilter volume
n_BF = 0.5                         # [g TAN / (m^3 BF · day)] coefficient
k_BF = 0.7                         # exponent
TAN0 = 0.1                         # initial TAN

#=========================================================================
# End of Parameters and constraints
#=========================================================================

#=========================================================================
# Equations Used - Define
#=========================================================================

def tao(t):         # effects of temperature tao(T)
 k = 4.6
 if t > T_opt:
  return math.exp(-k*((t - T_opt) / (T_max - T_opt))**4) 
 if t < T_opt:
  return math.exp(-k*((T_opt - t) / (T_opt - T_min))**4) 
 return 1.0

def segma(x):       # dissolved oxygen function segma(DO)
 if x > DO_crt:
  return 1
 if DO_min < x < DO_crt:
  return (x - DO_min)/(DO_crt - DO_min)
 else:
  return 0
  
def v(x):           # unionized ammonia v(UIA)
 if x < UIA_crit:
  return 1
 if UIA_crit < x < UIA_max:
  return (UIA_max - x) / (UIA_max - UIA_crit)
 else:
  return 0
 
def BF_capacity(TAN_prev):
    # BF_{t-1} = n_{t-1} * (TAN_{t-1})^{k-1}
    return n_BF * (TAN_prev ** k_BF)

def update_TAN(TAN_prev, feed_kg):
    feed_g = feed_kg * 1000.0     # convert to grams

    input_term   = (Nf * feed_g) / V_water          
    BF_prev      = BF_capacity(TAN_prev)            
    removal_term = (BF_prev * V_BF) / V_water       

    TAN_next = TAN_prev + input_term - removal_term
    return max(TAN_next, 0.0)  

def TAN_to_UIA(TAN, frac_UIA=0.05):
  # Assume 5% of TAN is unionized NH3
  return TAN * frac_UIA

def dwdt(t, w, UIA, f):      
 # Our main function describing the change in weight as a function of current weight and time dw/dt
 return h*p*f*b*(1-a)*tao(T)*segma(DO)*v(UIA)*w**m -k_min * math.exp(j*(T-T_min))*w**n

def profit(w_final ,w_initial, f):       
 # w is the final weight of the fish, f (in kg) is the feed in each day(list)
 total_feed = 0
 for i in range(len(f)):
  total_feed += f[i]
 return (w_final - w_initial) * p_fish - total_feed * p_feed

#=========================================================================
# Equations Used End
#=========================================================================

#=========================================================================
# Main Function
#=========================================================================

t_span      = [0, 1]                  # to integrate over 1 day
total_days  = 200
weights     = [w0]
feeds       = []
profit_lst  = []
feeding_list= np.arange(0.01,0.2,0.005)
pred_horiz  = 7
tan_lst     = [TAN0]

for day in range(total_days):
    ''' 
    We iterate over the total days of the simulation, each day, we calculate over a prediction horizen
    and predict the feed to get the max profit over it, once we find the feed, we choose it and add the 
    chosen parameters, and we continue to the next day.
    '''
    w_current    = weights[-1]
    pred_profit  = []
    TAN_current  = tan_lst[-1]

     # --- MPC prediction: try each candidate feed ratio ---
    for feed in feeding_list:
      pred_weights = [w_current]
      pred_feeds_kg = []
      w_tmp = w_current
      pred_tan = [TAN_current]
      tan_tmp = TAN_current
      valid = True

      # We iterate over the prediction horizen to check best profit 
      for pred_day in range(pred_horiz):
        UIA_current  = TAN_to_UIA(tan_tmp)

        # --- Firstly we check if the current UIA fits the constraints ---
        if UIA_current > UIA_crit:      
            valid = False
            break
        
        # --- We integrate over the next day of the prediction ---
        w_new_pred = solve_ivp(lambda t, w: dwdt(t, w, UIA_current, feed),t_span,[w_tmp])

        # --- Update Parameters: ---
        w_tmp = w_new_pred.y[0, -1]
        pred_weights.append(w_tmp)

        tan_tmp = update_TAN(tan_tmp, feed * w_tmp)
        pred_tan.append(tan_tmp)

        pred_feeds_kg.append(feed * w_tmp)

      if valid:
        pred_profit.append(profit(pred_weights[-1],pred_weights[0],pred_feeds_kg))
      else:
        # strongly penalize this feed so MPC will not choose it
        pred_profit.append(-1e9)
      
    
    # --- find the max profit and choose that
    max_index = int(np.argmax(pred_profit))
    max_feed  = feeding_list[max_index]
    feeds.append(max_feed)

    # recompute UIA from the actual TAN state
    UIA_real = TAN_to_UIA(TAN_current)
    w_new   = solve_ivp(lambda t, w: dwdt(t, w, UIA_real, max_feed),t_span,[w_current]) 
    w_next = w_new.y[0, -1]
    
    # --- we update daily params
    feed_today_kg = max_feed * w_current
    TAN_current = update_TAN(TAN_current, feed_today_kg)
    daily_profit = profit(w_next, w_current, [feed_today_kg])

    weights.append(w_next)

    tan_lst.append(TAN_current)

    if profit_lst:
      profit_lst.append(profit_lst[-1] + daily_profit)
    else:
      profit_lst.append(daily_profit)

    # ---break if the profit becomes negative --- loosing money
    #if daily_profit < 0:
      #break
  

#=========================================================================
# End of Main Function
#=========================================================================

# ========================================================================
# Plots
# ========================================================================

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
ax3.plot(range(len(feeds)), feeds, label='Feed Given', color='red')
ax3.set_ylabel('Feed (kg)')
ax3.set_xlabel('Time (days)')
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

# ========================================================================
# End Plots
# ========================================================================