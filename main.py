"""
Model Predictive Control (MPC) with Genetic Algorithm (GA)

@author: Orwa Kblawe
Date: Nov 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import params
#from params import *
from mpc import MPC
import bio_model
#from bio_model import TAN_to_UIA, run_sim

# =========================================================================
# Main
# =========================================================================


'''
DO_lst   = np.arange(DO_min, DO_max, 0.1)
Temp_lst = np.arange(T_min, T_max, 0.5)
feed_lst = np.arange(feed_min, feed_max, 0.0025)
Q_lst    = np.arange(Q_water_min, Q_water_max, 0.05)

w0   = 0.005    # initial weight in kg

total_days = 50
pred_horiz = 15
population_size = 50
num_gens = 50  

weights, feeds, applied_plan, profit_lst, tan_lst, no3_lst, co2_lst, feed_prec = MPC(
    total_days,
    pred_horiz,
    feed_lst,
    Temp_lst,
    DO_lst,
    Q_lst,
    w0,
    TAN0,
    population_size,
    num_gens,
    stop=False
)

# =========================================================================
# Plots 
# =========================================================================

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

# 1. Weight Plot
ax1.plot(range(len(weights)), weights, label='Weight', color='blue')
ax1.set_ylabel('Weight (kg)')
ax1.set_title(f'Simulation Results ({total_days} Days, feed price:{p_feed}, fish price: {p_fish})')
ax1.grid(True)
ax1.legend()

# 2. Profit Plot
ax2.plot(range(len(profit_lst)), profit_lst, label='Cumulative Profit', color='green')
ax2.set_ylabel('Profit ($)')
ax2.grid(True)
ax2.legend()

# 3. Feed Plot
ax3.plot(range(len(feeds)), feeds, label='Feed (kg/day)',color='red')
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

# 5. NO3 
ax5.plot(range(len(no3_lst)), no3_lst, label='Nitrate (mg/L)', color='brown')
ax5.plot([0, len(no3_lst)], [NO3_crit, NO3_crit], 'r--', label='Critical Limit')
ax5.set_ylabel('Nitrate (mg/L)')
ax5.set_xlabel('Time (days)')
ax5.grid(True)
ax5.legend()

# 6. CO2
ax6.plot(range(len(co2_lst)), co2_lst, label='Carbon Dioxite (mg/L)', color='orange')
ax6.plot([0, len(co2_lst)], [CO2_crit, CO2_crit], 'r--', label='Critical Limit')
ax6.set_ylabel('CO2 (mg/L)')
ax6.set_xlabel('Time (days)')
ax6.grid(True)
ax6.legend()

plt.tight_layout()
plt.show()

# =========================================================================
# End Plots
# =========================================================================
# =========================================================================
# Calibration & Validation
# =========================================================================
'''

DO_const = 6
Q_const = 0.6

T_arr = [24]*70 + [27]*126
Data_days = [0, 2, 22, 29, 36, 56, 63, 78, 84, 91, 105, 112, 120, 155, 168, 177, 190, 196]

# ==== Tuning parameters =====

bio_model.h = 0.5      #  
bio_model.p = 1.5      
bio_model.b = 0.62
bio_model.a = 0.53             
bio_model.m = 0.9     #
bio_model.n = 0.7    #
bio_model.k = 1.2      #

# ==== Tank A - Calibration ===============================================
#'''
N_fish = 66

Feed_arr = [70]*23 + [115]*55 + [129]*6 + [117]*7 + [187]*14 + [124]*7 + [131]*8 + [168]*35 + [208]*13 + [160]*9 + [161]*13 + [168]*6

Data = [14, 14.2, 16, 17.35, 17.9, 20.84, 20.68, 28.8, 25.8, 26, 30, 32, 33.2, 42, 39.7, 40.4, 43, 41.7]
#'''
# ==== Tank B - Validation =================================================
'''
N_fish = 40

Feed_arr = [100]*23 + [185]*55 + [201]*6 + [187]*7 + [52]*14 + [57.16]*7 + [53]*8 + [62]*35 + [63]*13 + [68]*9 + [70]*13 + [70]*6

Data = [49.8, 61, 61.8, 64.4, 72.1, 75.85, 75.4, 84.1, 78.2, 80, 88.5, 78.5, 86, 89.5, 93.25, 97, 96.9, 98.9]

#'''
# ==== Tank D - Validation =================================================
'''
N_fish = 40

Feed_arr = [100]*23 + [185]*55 + [214]*6 + [217]*7 + [320]*14 + [182]*7 + [181]*8 + [235]*35 + [255]*13 + [252]*9 + [255]*13 + [257]*6

Data = [59.2, 57.1, 61.8, 67.7, 70, 76.5, 77.6, 89.27, 90.7, 100, 91.6, 91.12, 98, 106, 103.4, 105, 107.2, 108.9]
#'''
# ==========================================================================

w_init = Data[0] / 1000.0

days = np.arange(196)
weights_measured = np.interp(days, Data_days, Data) 

feed_ratios = [Feed_arr[i] / (weights_measured[i] * N_fish )for i in range(len(weights_measured))]

weights_measured = []
i = 0
for k in range(len(Data_days)-1):
    days = Data_days[k+1] - Data_days[k]
    for _ in range(days):
        weights_measured.append(Data[i])
    i += 1

W_est = np.array(weights_measured[:196], dtype=float) / 1000.0

feed_total_kg_per_fish = (np.array(Feed_arr, dtype=float) / 1000.0) / N_fish
feed_ratio = feed_total_kg_per_fish / W_est   

ind = [(float(feed_ratio[d]), float(T_arr[d]), float(DO_const), float(Q_const)) for d in range(196)]

weights, feeds_kg, tan_lst, no3_lst, co2_lst, temps, DOs, Q_waters = bio_model.run_sim(ind, w_init, params.TAN0)

weights = np.array(weights, dtype=float)   
days_sim = np.arange(len(weights))
calibrated= days_sim          
Data_g = np.array(Data, dtype=float)
Data_kg = Data_g * 1000.0

# ===== The Stock System =====

bio_model.h  = 0.8        
bio_model.p  = 1.5
bio_model.b  = 0.62
bio_model.a  = 0.53             
bio_model.m  = 0.67
bio_model.n  = 0.81
bio_model.k  = 4.6

weights_old, feeds_kg, tan_lst, no3_lst, co2_lst, temps, DOs, Q_waters = bio_model.run_sim(ind, w_init, params.TAN0)
weights_old = np.array(weights_old, dtype=float) 
# ===========================

plt.figure(figsize=(10,6))
plt.plot(calibrated, weights * 1000, label="Calibrated Simulated weight (g)")
plt.plot(calibrated, weights_old * 1000,'r--', label="Non-Calibrated Simulated weight (g)")
plt.plot(Data_days, Data_g, "o", label="Measured mean weight (g)")
plt.xlabel("Day")
plt.ylabel("Mean fish weight (g)")
plt.title("Simulated vs Measured")
plt.grid(True)
plt.legend()
plt.show()


# =========================================================================
# End Calibration & Validation
# =========================================================================