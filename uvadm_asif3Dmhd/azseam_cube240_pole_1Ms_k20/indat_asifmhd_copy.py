# Input File for ADM_FULL

#### IMPORT REQUIRED PACKAGES ####
import numpy as np

#### MODEL FOLDER NAME ####
model_directory = 'azseam_cube240_pole_1Ms_k20'
#model_directory = 'azseam_cube240_eq_1Ms_k20'
#model_directory = 'cube200'

#### MHD SIM DATACUBE NAME ####
#datacube = 'for_christi_cube100.dat'
#datacube = 'cube200_T1oc_1Ms.dat'

#### USEFUL VARIABLES ####

nx=240
ny=240
nz=240

# Stellar Parameters

# cube100
#Rstar = 1.322e12 #cm
#Vinf = 3000 # km/s
#Bpole = 2950 #G
#Mdot = 3.2e-6 # Msun/yr

# cube200
#Rstar = 8 * 7.e10 #cm
#Vinf = 3200 # km/s
#Bpole = 1100 #G
#Mdot = 3.3e-7 # Msun/yr

vth = 0.01 #vinf

#kap = 0.1
#kap = 0.5
#kap = 1.0
#kap = 5.0
#kap = 10.0
kap = 20.0

lmax=1.7
dl=0.05


