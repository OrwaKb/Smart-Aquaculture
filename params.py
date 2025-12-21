

# =========================================================================
# Parameters and constraints
# =========================================================================

h        = 0.8
p        = 1.5
b        = 0.62
a        = 0.53
m        = 0.67
n        = 0.81

T_min    = 24
T_opt    = 33   # intrinsic optimal temperature used in tao(T)
T_max    = 40
T_env    = 28   # Temperature of the room

k_min    = 0.00133
j        = 0.0132

DO_crt   = 3.0
DO_min   = 1.0
DO_max   = 15.0
DO_base  = 0.5    # Baseline DO
 
UIA_crit = 0.06
UIA_max  = 1.4

p_feed   = 0.75   # feed price per kg
p_fish   = 2.0    # fish price per kg

feed_min = 0.01
feed_max = 0.3

# TAN / biofilter parameters
TAN0     = 0.0    # initial TAN
PC       = 0.6    # Protien content Page 642
NPU      = 0.5    # from yellow book page 77
Nf       = 0.03   # [g TAN / g feed]
V_water  = 1.0    # [m^3] system water volume 
V_BF     = 0.002   # [m^3] biofilter volume was 0.1
n_BF     = 0.5    # [g TAN / (m^3 BF · day)] was 1.3
k_BF     = 0.7    # exponent

area     = 10.5  # [m2] the total area of heat transfere. calculated based on the system in the lab  

PH       = 7     # level of PH in the water (7)
PK_a     = 9.4   # acidity constant 9.4 at 20C

electricity_price = 0.2 #0.00015 

N_fish  = 100 #40

TAN_max = 1.0

NO3_0    = 0.0    # Initial Nitrate concentration
NO3_crit = 150.0  # Critical limit for Nitrate

#=============================================================================