"""
Model Predictive System Part

Orwa Kblawe 

Nov 2025
"""


#Parameters and constraints

P_fi = 1      
P_fe = 1

w_o   = 10

T_s = NaN
p = NaN
m = NaN

"""
P_fi is the price of the fish (per kg)
P_fe is the price of the feed (per kg)
w_0 is the initial weight of the fish (in kg)
T_s is the sample time (in ??) recomendation to fit 10 - 20 samples within the raise time of the open loop response
p is the predictive horizon recommendation to fit 20 - 30 samples covering the open-loop transient system response
m is the control horizon recommendation is 10 - 20% of the prediction horizon and a minimum of 2-3 steps
include hard and soft constraints here...
weights may be included to emphasize the more important params
"""

