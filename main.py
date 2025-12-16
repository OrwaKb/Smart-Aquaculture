"""
Model Predictive Control (MPC) with Genetic Algorithm (GA)
for optimal temperature and dissolved oxygen selection.

@author: Orwa Kblawe
Date: Nov 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from params import *
from mpc import MPC
from bio_model import TAN_to_UIA

# =========================================================================
# Main
# =========================================================================

DO_lst   = np.arange(DO_min, DO_max, 0.1)
Temp_lst = np.arange(T_min, T_max, 0.5)
feed_lst = np.arange(feed_min, feed_max, 0.0025)

TAN0 = 0.01    # initial TAN
w0   = 0.05    # initial weight in kg

total_days = 200
pred_horiz = 7

weights, feeds, applied_plan, profit_lst, tan_lst = MPC(
    total_days=30,
    pred_horiz=7,
    feeding_lst=feed_lst,
    T_lst=Temp_lst,
    DO_lst=DO_lst,
    initial_weight=w0,
    initial_Tan=TAN0,
    population_size=30,
    num_gens=10,
    stop=False
)

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
ax3.plot(feeds, label='Feed (kg/day)',color='red')
ax3.set_ylabel('Feed (kg/day)')
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
