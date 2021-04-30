## C. Erba, V. Petit
## UV ADM RT, Version 5.0
## Module Containing Functions for adm_full.py
## In original form: 7/30/19
## Last Updated: 10/10/19

###########################
# IMPORT REQUIRED PACKAGES #
############################

import numpy as np

####################
# INPUT PARAMETERS #
####################

####
# Beta-Law (Velocity) Enhancement Factor
# Affecting Upflow Velocities
####

#vifac = 1.5
vifac = 1.0
v_offset = 0.5 # In v_thermal units, so that v is not zero at r=0.
vesc_vinf = 1.0/3.0
#vesc_vinf = 1.0/2.6

####
# Import Stock Files for Shock Retreat Calculations
####

mushock_transceq=np.genfromtxt('shock_retreat_parameters/mu_shock.dat', skip_header=1, unpack=True)
rshock_transceq=np.genfromtxt('shock_retreat_parameters/r_shock.dat', skip_header=1, unpack=True)

def mu_shock(chii_value):
    if chii_value == 0.01: return mushock_transceq[0]
    elif chii_value == 0.1: return mushock_transceq[1]
    elif chii_value == 1.0: return mushock_transceq[2]
    elif chii_value == 10.0: return mushock_transceq[3]
    elif chii_value == 100.0: return mushock_transceq[4]
    else: return print("Error in mu_shock: Incorrect value of Chi_infinity chosen. Please check indat file.")

def r_shock(chii_value):
    if chii_value == 0.01: return rshock_transceq[0]
    elif chii_value == 0.1: return rshock_transceq[1]
    elif chii_value == 1.0: return rshock_transceq[2]
    elif chii_value == 10.0: return rshock_transceq[3]
    elif chii_value == 100.0: return rshock_transceq[4]
    else: return print("Error in r_shock: Incorrect value of Chi_infinity chosen. Please check indat file.")

#################
# ADM FUNCTIONS #
#################

####
#Magnetic Moment Vector
# Returns magnetic moment vector of a given point
# Third line of the rotation matrix from LOS->B
# Rotation around X_LOS
####

def mvbp(aa): return np.array([0.0,-np.sin(aa),np.cos(aa)])

####
# ADM mu’s
# mu_b = mdot_b(mu_star): ADM EQ. 6
####

def mu(r_hat, mvbp):
    # r_hat is a (nz,3) array
    mu = np.zeros(r_hat.shape[0])
    for i in range(mu.size):
        mu[i] = np.dot( mvbp,r_hat[i,:] )
    return(np.abs(mu))

def mu_st_sq(r,mu): return 1 - (1 - mu**2)/r

def mu_b( mu_st_sq ):
    return( 2.0 * mu_st_sq**0.5 / (1.0 + 3.0 * mu_st_sq)**0.5 )

####
#Max radius of loop closure (r_m)
####

def rm(mu_st_sq): return 1.0/(1.0-mu_st_sq)

####
#Magnetic Field Parameters
#Bhatz: direction of magnetic field along z_los
#B_Bstar: ADM EQ 3
####

def bhatz(mvbp, r_hat, r):
    # r_hat is a (nz,3) array
    Bhat_xyz = np.zeros(r_hat.shape)
    vup_sign = np.zeros(r_hat.shape[0])

    for i in range(0,r_hat.shape[0]):
        Bhat_xyz[i,:] =  ( 3 * r_hat[i,:] * np.dot(mvbp,r_hat[i,:]) - mvbp ) / r[i]**3
        Bhat_xyz[i,:] = Bhat_xyz[i,:] / ( Bhat_xyz[i,0]**2 + Bhat_xyz[i,1]**2 + Bhat_xyz[i,2]**2 )**0.5
        vup_sign[i] = np.sign(np.dot( Bhat_xyz[i,:], r_hat[i,:] )) # Also calculate the sign for the velocity, while we are here.
    return(Bhat_xyz[:,2], vup_sign) # Only return the z component of B hat

def B_Bstar(r, mu, mu_st_sq):
    return( r**(-3) * (1.0+3.0*mu**2)**0.5 / (1.0+3.0*mu_st_sq)**0.5 )

####
# Functions calculating parameters needed 
# to describe Velocity, Density
# of Upflow, Downflow material
####

def up( r, B_Bstar, mu_st_sq, bhatz, vtherm, sign, open_up ):
    ## magnitude of velocity; beta-law with rstar and beta = 1
    wu = (1.0 - 1.0/r) + v_offset*vtherm
    rhoup = B_Bstar * mu_b(mu_st_sq) / wu
    ## Return the signed vz_up, and the density
    vz = sign*wu*bhatz
    #############
    ## This applies the vifac if wanted:
    ####
    ## In the open loops only:
    #vz[open_up] = vz[open_up]*vifac
    #rhoup[open_up] = rhoup[open_up]/vifac
    ####
    ## In open+closed loops:
    #vz = vz*vifac
    #rhoup = rhoup / vifac #Christi, mass continuity
    #############
    return(vz, rhoup)

def down(r, B_Bstar, mu_st_sq, bhatz, d, sign, mu):
    ## Note, the sign for north/south loops is calculated in the main code.
    ## Negative sign for direction of downflow velocity is in the main code.
    wd = vesc_vinf * np.abs(mu) * r**(-0.5)
    rhod = B_Bstar * mu_b(mu_st_sq) * np.sqrt(r) / ( mu**2 + ( d**2/r**2) )**0.5 / vesc_vinf
    # Return the signed vz_dn, and the density
    return( sign*wd*bhatz, rhod )



######################
# EMISSION FUNCTIONS #
######################

####
# Basic Equations for Coding the Emission Part of the P-Cygni Profiles
#
# Function for setting boundary conditions (in p)
## Vero: We consider that if a ray has p==1, then
## it intersects the stellar disk.
## The beta law velocity is pegged by 0.25 v_thermal
## so that the density doesn't blow up.
## Thus there is no need for fancy boundary conditions.
####

def pbound(xx,yy,z_end,z_s):
    if np.sqrt(xx**2+yy**2)<=1:
        return np.arange(np.sqrt(1-(xx**2+yy**2)),z_end+z_s,z_s)
    else:
        return np.arange(-z_end,z_end+z_s,z_s)

def sthin(r):
    W = (1.0 - (1.0/r)**2 )**0.5 #OR85 "mustar"
    return( (1.0 - W)/2.0 )


#######################
# SPHERICAL FUNCTIONS #
#######################

#def spherical(r, rhat):
#    wd = vesc_vinf * np.abs(mu) * r**(-0.5)
#    return(wd*r_hat[:,2], )

