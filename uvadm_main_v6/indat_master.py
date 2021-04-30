# Input File for ADM_FULL

#### IMPORT REQUIRED PACKAGES ####
import numpy as np

#### MODEL FOLDER NAME ####

#model_directory = 'a0_kp5_x1_ra2p7_dp1'
model_directory = 'test'

#### DEFINE GRID COORDINATES ####

dx=0.05
dy=0.05
dz=0.05

#xmax=10.0
xmax=10.0
ymax=10.0
zmax=10.0

#### MODEL PARAMETERS ####
vth = 0.01
# set vifac in module

# chii value must be one of [0.01,0.1,1.0,10.0,100.0]
# or, recalculate new stock r_shk, mu_shk from transcendental eq
# using ‘shock_retreat_calculate_boundaries’ notebook

#chii = 100.0
#chii = 10.0
chii = 1.0
#chii = 0.01

#r_alfv = 2.7
r_alfv = 10.0

delta = 0.1

# ALPHA NEEDS TO BE IN RADIANS!
alp = np.radians(90.0)

#kap = 0.1
#kap = 0.5
kap = 1.0

lmax=1.2
dl=0.05


