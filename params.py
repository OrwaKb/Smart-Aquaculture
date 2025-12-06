

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
DO_max   = 5.0
DO_base  = 2.0   # Baseline DO
 
cT       = 0.00  # cost per celcious * day                    #
cDO      = 0.03  # cost per mg/l         

UIA_crit = 0.06
UIA_max  = 1.4

p_feed   = 0.7  # feed price per kg
p_fish   = 2.0  # fish price per kg

# TAN / biofilter parameters
Nf       = 0.03   # [g TAN / g feed]
V_water  = 100.0  # [m^3] system water volume
V_BF     = 10.0   # [m^3] biofilter volume
n_BF     = 0.5    # [g TAN / (m^3 BF · day)]
k_BF     = 0.7    # exponent

# =========== Parameters from m.saadi's code ===========

area     = 10.5 # [m2] the total area of heat transfere. calculated based on the system in the lab  

PH       = 7     # level of PH in the water (7)
PK_a     = 9.4   # acidity constant 9.4 at 20C

electricity_price = 0.06 

#=============================================================================