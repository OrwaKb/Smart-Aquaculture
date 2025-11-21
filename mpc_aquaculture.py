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
p     = 1.5         #
b     = 0.62
a     = 0.53
m     = 0.67
n     = 0.81
T     = 30          #
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
UIA   = 0.5
p_feed= 0.7                      # feed price per kg
p_fish= 2                        # fish price per kg
w0    = 0.1                        # initial weight in kg

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

def segma(x):       # dissolved oxygen function segma(DO)
 if x > DO_crt:
  return 1
 
 if DO_min < x < DO_crt:
  return (DO - DO_min)/(DO_crt - DO_min)
 
 else:
  return 0
  
def v(x):           # unionized ammonia v(UIA)
 if x < UIA_crit:
  return 1
 
 if UIA_crit < x < UIA_max:
  return (UIA_max - UIA) / (UIA_max - UIA_crit)
 
 else:
  return 0

def dwdt(t, w, f):      
 # Our main function describing the change in weight as a function of current weight and time dw/dt

 return h*p*f*b*(1-a)*tao(T)*segma(DO)*v(UIA)*w**m -k_min * math.exp(j*(T-T_min))*w**n

def profit(w, f):       
 # w is the final weight of the fish, f (in kg) is the feed in each day(list)
 total_feed = 0
 for i in range(len(f)):
  total_feed += f[i]
 return w * p_fish - total_feed * p_feed

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
feeding_list= np.arange(0.001,0.06,0.001)

'''
Here is the Main function, for now this calculate the best feed out of (0.1,0.2,...,1) precent out of the fish weight,
and when it finds the one with the best profit it chooses that and continues to the next day - meaning out prediction horizen = 1 for now
'''

for day in range(total_days):
    days_weights = []
    days_list = []
    w_current = weights[day]              
    for feed in feeding_list:
        w_new   = solve_ivp(dwdt, t_span, [w_current], args=(feed,))
        days_weights.append(w_new.y[-1][-1])
        days_list.append(feed * w_current)

    days_profit = []
    for i in range(len(days_weights)):
        days_profit.append(profit(days_weights[i],[days_list[i]]))

    # Figure out which feed gives max profit
    info_mat = np.column_stack((days_list,days_weights ,days_profit))
    last_column = info_mat[:, -1]
    max_index = np.argmax(last_column)
    best_row = info_mat[max_index]
    
    # Add everything and update current weight
    weights.append(best_row[1])
    feeds.append(best_row[0])          # This calculates the feed per kg (f is the feed pecent from the fish weight, w is the weight of the day)
    profit_lst.append(best_row[2])


#=========================================================================
# End of Main Function
#=========================================================================

# ========================================================================
# Plots
# ========================================================================



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

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

plt.tight_layout()
plt.show()

# ========================================================================
# End Plots
# ========================================================================