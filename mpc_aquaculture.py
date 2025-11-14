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
f     = 0.9         #
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
UIA   = 1

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

def dwdt(w, t):      # Our main function describing the change in weight as a function of current weight and time dw/dt
 return h*p*f*b*(1-a)*tao(T)*segma(DO)*v(UIA)*w**m -k_min * math.exp(j*(T-T_min))*w**n

#=========================================================================
# Equations Used End
#=========================================================================

'''
result = sp.integrate(dwt, [0,1], [1])
print(result)
x = []
y = []
'''
"""
for i in np.arange(0,1,0.1):
  x.append(i)
  y.append(dwt(i))
 
# Plot
plt.figure()
plt.plot(x, y)
plt.xlabel("weight")
plt.ylabel("dw(t) / dt")
plt.title("S")
plt.grid(True)

plt.show()
"""
#=========================================================================
# Main Function
#=========================================================================

'''
Put initial weight and integrate over one day using the equation of the fish
growth model 
'''

w0 = 0.1                         # initial weight in kg
t_span = [0, 1]                  # integrate over 1 day
t_eval = np.linspace(0, 1, 100)  # for smooth curve

sol = [w0]

total_items = 200
total_time  = []
weights     = []

for i in range(total_items):
 w_final = sol[i]
 w_new   = solve_ivp(dwdt, t_span, [w_final], method='RK45')
 total_time.append(w_new.t + i)
 weights.append(w_new.y)
 sol.append(w_new.y[0, -1])

weights_stack = np.hstack(weights)
time_stack = np.hstack(total_time)
print(sol[total_items])

#=========================================================================
# End of Main Function
#=========================================================================

# ========================================================================
# Plots
# ========================================================================

import matplotlib.pyplot as plt

plt.plot(time_stack, weights_stack[0], label='Weight (kg)')
plt.xlabel('Time (days)')
plt.ylabel('Weight (kg)')
plt.title('Fish Growth Over 1 Day')
plt.legend()
plt.grid(True) 
plt.show()

# ========================================================================
# End Plots
# ========================================================================