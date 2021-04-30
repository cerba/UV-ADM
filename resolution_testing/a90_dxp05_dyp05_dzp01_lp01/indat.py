# Input File for ADM_FULL

#### IMPORT REQUIRED PACKAGES ####
import numpy as np

#### MODEL FOLDER NAME ####

model_directory = 'a90_dxp05_dyp05_dzp01_lp01' 

#### DEFINE GRID COORDINATES ####

dx=0.05
dy=0.05
dz=0.01

xmax=10.0
ymax=10.0
zmax=10.0

#### MODEL PARAMETERS ####
vth = 0.01
# set vifac in module

# chii value must be one of [0.01,0.1,1.0,10.0,100.0]
# or, recalculate new stock r_shk, mu_shk from transcendental eq
# using ‘shock_retreat_calculate_boundaries’ notebook
chii = 0.01

r_alfv = 2.7
#r_alfv = 10.0

delta = 0.1
#delta = 0.001

# ALPHA NEEDS TO BE IN RADIANS!
alp = np.radians(90.0)
#alp = np.radians(45.0)

kap = 1.0

lmax=1.7
dl=0.01


