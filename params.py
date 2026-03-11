

# =========================================================================
# Parameters and constraints
# =========================================================================

h        = 0.6        
p        = 1.5
b        = 0.62
a        = 0.53             
m        = 0.87
n        = 0.78

k        = 4.6

T_min    = 24
T_opt    = 33         # intrinsic optimal temperature used in tao(T)
T_max    = 40
T_env    = 30         # Temperature of the room

k_min    = 0.00133
j        = 0.0132    

DO_crt   = 1.0
DO_min   = 0.3
DO_max   = 15.0
DO_base  = 0.5        # Baseline DO
 
UIA_crit = 0.06
UIA_max  = 1.4

p_feed   = 0.5        # feed price per kg
p_fish   = 8.0        # fish price per kg

feed_min = 0.01
feed_max = 0.3

# TAN / biofilter parameters
TAN0     = 0.0        # initial TAN
PC       = 0.4        # Protien content Page 642
NPU      = 0.5        # from yellow book page 77
#Nf       = 0.03      # [g TAN / g feed]
V_water  = 2.3        # [m^3] system water volume 
V_BF     = 0.25 * V_water  # [m^3] biofilter volume, Dr.Alaa "biofilter volume should be aroung 1/4 of the tank volume"
n_BF     = 1.3        # [g TAN / (m^3 BF · day)] was 300, 1.3
k_BF     = 0.7        # exponent

area     = 2          # [m2] the total area of heat transfere. calculated based on the system in the lab is 10.5  

PH       = 7.5        # level of PH in the water (7)
PK_a     = 9.4        # acidity constant 9.4 at 20C

electricity_price = 0.2 #0.00015 

# =========== From Dr.Alaa - "we should have up to 20kg/m3 fish in tank" ===========

rho_fish = 20.0
W_avg    = 0.15

N_fish  = (rho_fish * V_water) / W_avg         # Number of fish in a tank

# ==================================================================================

TAN_max = 1.0

NO3_0    = 0.0        # Initial Nitrate concentration
NO3_crit = 150.0      # Critical limit for Nitrate

CO2_0    = 0.5
CO2_crit = 20.0       # mg/L, recommend P47, else Bohr-Root effect
CO2_max  = 40.0       # fish dies

Alk      = 150.0      # mg/L as CaCO3 (Assumed constant)

# =========== Water exchange values ===========

T_fresh      = 20.0   # [C] Cold tap water
NO3_fresh    = 0.0    # [mg/L] Clean
CO2_fresh    = 2.0    # [mg/L]
TAN_fresh    = 0.0

k_strip      = 5.0    # day^-1
p_water      = 0.005  # [$/m3] Water price
Cp_water     = 4186   # [J/kg/C] #???
rho_water    = 1000   # [kg/m3]

# Constraints
Q_water_min  = 0.0    # [m3/day]
Q_water_max  = 0.5    # [m3/day] Exchange up to 500L/day

#=============================================================================