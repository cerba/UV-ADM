## C. Erba, V. Petit,
## UV ADM RT, Version 5.0
## In original form: 7/30/19
## Last Updated: 10/19/20

#######################
#### VERSION NOTES ####
#######################

# Update 11/5/19 adds new good_up condition:
# good_up =  (np.unique(np.append( open_up[0], good_up[0] )) ,) # Make a new good_up that includes the open field line upflow
# This ensures we check two conditions: inside/outside Alfven radius & inside/outside shock radius

# Update 10/19/20 adds Stokes V capability

#######################################
#### SET UP DIRECTORY, INDAT FILES ####
#######################################
import os, sys, inspect, pathlib, scipy, time
from shutil import copyfile
import indat_master as ind_m

filepath = ind_m.model_directory
print('Model Directory is:', filepath)
pathlib.Path(filepath).mkdir(exist_ok=False) #Make new model directory; will not create if directory already exists

copyfile('indat_master.py', filepath+'/indat.py') #copy indat file into new directory; use copied indat for calculations below
copyfile('adm_basecode_master.py', filepath+'/adm_basecode.py') #copy basecode file into new directory; use copied basecode for calculations below

##################################
#### IMPORT REQUIRED PACKAGES ####
##################################

import numpy as np
from scipy import interpolate
from scipy.special import erf
import matplotlib.pyplot as plt

sys.path.insert(0, filepath)
import indat as ind
import adm_basecode as adm
## Check that we are using the right indat/basecode files
print('Indat file is:', inspect.getfile(ind) )
print('Basecode file is:', inspect.getfile(adm) )
print('Indat files read.')

# IMPORT ASCII WRITER/READER
from astropy.io import ascii

# START THE CLOCK
start = time.time()

#############################################
#### CALCULATE GRID, BOUNDARY CONDITIONS ####
#############################################

#### CALCULATE X,Y GRID ####
xlist=np.arange(-ind.xmax,ind.xmax+ind.dx,ind.dx)
ylist=np.arange(-ind.ymax,ind.ymax+ind.dy,ind.dy)
nx = xlist.size
ny = ylist.size

A = ind.dx * ind.dy 

#### Array of doppler shifts, in velocity
#### Note, redshift is negative velocity.
lamlist=np.arange(-ind.lmax,ind.lmax+ind.dl,ind.dl)
#lamlist=np.array([0.05]) #positive is blue side
nlam = lamlist.size

print('Numbers of rays: {}'.format(nx*ny))
print('Numbers of elements in x, y, and lambda: {}'.format(nx*ny*nlam))

iabs = np.zeros( (nx, ny, nlam) )
iemit = np.zeros( (nx, ny, nlam) )
intensity_antiV = np.zeros( (nx, ny, nlam) )

print('Size of each result matrix is {} Mbytes'.format(iabs.nbytes/1e6))

# Normalization Constant
xx, yy = np.meshgrid(xlist, ylist)
pp = (xx**2 + yy**2)**0.5
n_disk = np.where(pp<=1.0)
xynorm = A * n_disk[0].size

print('Normalization constant is {} Pi'.format(xynorm/np.pi))

######################################
#### MAIN RADIATIVE TRANSFER LOOP ####
######################################

print('Starting main loop calculation.')

#### To make plots “on the fly,” initialize these:
#fig, ax = plt.subplots(1,1, figsize=(7,7))
#ax.set_aspect('equal')
####

# Step 0a: Calculate cubic spline function using rs, mus from "stock"
cspl = interpolate.CubicSpline(adm.mu_shock(ind.chii),adm.r_shock(ind.chii))

# Step 0b: Calculate Magnetic Moment
mvbp = adm.mvbp(ind.alp)

for i, x in enumerate(xlist): #all rays
##for i, x, in enumerate( np.array([1.1])): # one ray; pick a single x; np.array([0]) is x=0 plane
    
    for j, y in enumerate(ylist): #all rays
    ##for j, y, in enumerate( np.array([2.0])): # one ray; pick a single y
    
        ## Step 1: Get the ray in z, r. Runs from the star to the observer.
        ## This means that delta_tau is negative (Dec to obs where Tau=0)
        #z_step = adm.pbound(x,y,ind.zmax,ind.dz)
        if np.sqrt(x**2+y**2)<=1:
            z_step = np.arange(np.sqrt(1.0-(x**2+y**2)),ind.zmax+ind.dz,ind.dz)
            r_list = np.append( np.array([1.0]), ( x**2 + y**2 + z_step[1:]**2 )**0.5 )
        else:
            z_step = np.arange(-ind.zmax,ind.zmax+ind.dz,ind.dz)
            r_list = ( x**2 + y**2 + z_step**2 )**0.5
        nz = z_step.size

        ## Step 2: Calculate the values of rhat and mu on the ray
        #toto = ax.scatter( z_step, [y]*(nz), c=r_list, edgecolor='none', s=5, vmin=0,vmax=5  )
        rhat_list = np.zeros( (nz, 3) )
        rhat_list[:,0]=x/r_list
        rhat_list[:,1]=y/r_list
        rhat_list[:,2]=z_step/r_list
        mu_list = adm.mu(rhat_list, mvbp)
	
	## Step 3: Boundary Conditions for inside/outside Shock, Alfven radius
        r_shock = cspl(mu_list) # Value of r at the shock boundary for our mus.
        mu_st_sq_list = adm.mu_st_sq( r_list, mu_list )
        rm_list = adm.rm(mu_st_sq_list)
        
        good_down = np.where( rm_list <= ind.r_alfv ) # Downflow only inside Alfven radius
        open_up = np.where( rm_list > ind.r_alfv ) # For the upflow -- is it an open loop?
        good_up = np.where( r_list <= r_shock ) # Is our current r less than r_shock for that mu?
        good_up =  (np.unique(np.append( open_up[0], good_up[0] )) ,) # Make a new good_up that includes the open field line upflow; added 11/5/19
        # Leave the comma in good_up; it's important 

	## Step 4: Calculate S_thin
        srfc_list = adm.sthin(r_list)
        #W = (1.0 - (1.0/r_list)**2 )
        #toto = ax.plot(z_step, srfc_list)
        
                
	## Step 5: Calculate B-field parameters, Velocity, Density
        bhatz_list, vup_sign = adm.bhatz(mvbp, rhat_list, r_list )
        B_Bstar_list = adm.B_Bstar( r_list, mu_list, mu_st_sq_list )
        vzu_list, rhou_list = adm.up( r_list, B_Bstar_list, mu_st_sq_list, bhatz_list, ind.vth, vup_sign, open_up)
        vzd_list, rhod_list = adm.down(r_list, B_Bstar_list, mu_st_sq_list, bhatz_list, ind.delta, -1*vup_sign, mu_list) 
        
        Bz_Bstar_list = B_Bstar_list * bhatz_list #Gives Bz / Bstar; Update--Stokes V
        B_Bpole_list = np.sqrt(3*mu_st_sq_list + 1 ) #Gives |Bstar| / Bpole; Update--Stokes V
        
        #toto = ax.plot(z_step, B_Bstar_list)
        #toto = ax.plot(z_step, Bz_Bstar_list)
        #toto = ax.plot(z_step, B_Bpole_list)
        
        # Plot One plane
        #toto = ax.scatter( z_step, [y]*(nz), c=vzu_list, edgecolor='none', s=5, vmin=-1.5,vmax=1.5, cmap='seismic_r' )
        #toto = ax.scatter( z_step, [y]*(nz), c=vzd_list, edgecolor='none', s=5, vmin=-0.3,vmax=0.3, cmap='seismic_r' )
        #Plot One Ray
        #toto = ax.plot(z_step, vzu_list)
        #toto = ax.plot(z_step, vzd_list)

        ## All values so far are calculated at the edges of the slab.
        ## thus my delta v, and delta tau arrays have one less element.
        ## If the left edge is in, then we consider it.
        
        dvzu_dz = np.abs(vzu_list[0:-1]-vzu_list[1:])/ind.dz
        dvzd_dz = np.abs(vzd_list[0:-1]-vzd_list[1:])/ind.dz

        # Plot One Plane
        #toto = ax.scatter( z_step[0:-1], [y]*(z_step.size-1), c=dvzu_dz, edgecolor='none', s=5, vmin=-1.5, vmax=1.5)
        #toto = ax.scatter( z_step[0:-1], [y]*(z_step.size-1), c=dvzd_dz, edgecolor='none', s=5, vmin=-0.5, vmax=0.5)
        # Plot One Ray
        #toto = ax.plot(z_step[0:-1], dvzu_dz)
        #toto = ax.plot(z_step[0:-1], dvzd_dz)

	## Step 6: Calculate Optical Depth
    
        small_mup = np.where( dvzu_dz[good_up[0][0:-1]] <= 0.001 ) # Condition for when slope is close to zero
        small_mup=small_mup[0]
        small_erfup = np.where( dvzu_dz[good_up[0][0:-1]] > 0.001 ) # Condition for when slope is close to zero
        small_erfup = small_erfup[0]
        
        delta_tau_up = np.zeros( (nz-1, nlam) ) # Define the delta-tau array for upflow
        # Here we are creating variables for the Taylor Expansion expr, so we don't have to rewrite them 100 times
        z1u = z_step[good_up[0][small_mup]]   #z_step, left side of slab, upflow
        z2u = z_step[good_up[0][small_mup]+1] #z_step, right side of slab, upflow
        mloc_up = dvzu_dz[good_up[0][small_mup]]
        #Calculate delta-tau-up:
        if good_up[0][0:-1].size > 0:
            for l, v in enumerate(lamlist): #enumerate defines i, x => index, variable
                # Calculation for the ERF
                delta_tau_up[good_up[0][small_erfup], l] = \
                    ind.kap*(0.5*(rhou_list[good_up[0][small_erfup]]+rhou_list[good_up[0][small_erfup]+1])) * (0.5 / dvzu_dz[good_up[0][small_erfup]]) \
                    * np.abs( erf( (v - vzu_list[good_up[0][small_erfup]])/ind.vth ) - erf( (v - vzu_list[good_up[0][small_erfup]+1])/ind.vth )  )
                # Calculation for the Taylor Expansion
                gammau = (v - vzu_list[good_up[0][small_mup]])/ind.vth #Define for simplicity
                delta_tau_up[good_up[0][small_mup], l] = \
                    ind.kap*(0.5*(rhou_list[good_up[0][small_mup]]+rhou_list[good_up[0][small_mup]+1]))* \
                    (-1/(3*np.sqrt(np.pi)*ind.vth**3)) * ( np.exp(-gammau**2) * (z1u-z2u) * ( 3*ind.vth**2 + 3*mloc_up*ind.vth*(z2u-z1u)*gammau + \
                    mloc_up**2 * (z1u-z2u)**2 * (2*gammau**2 - 1) ) )

        # Plot One Plane
        #toto = ax.scatter( z_step[0:-1], [y]*(z_step.size-1), c=delta_tau_up[:,15], edgecolor='none', s=5, vmin=-0.001,vmax=0.001, cmap="Blues"  )
        # Plot One Ray
        #toto = ax.plot(z_step[0:-1], delta_tau_up[:,15])

        small_mdown = np.where( dvzd_dz[good_down[0][0:-1]] <= 0.001 ) # Condition for when slope is close to zero
        small_mdown=small_mdown[0]
        small_erfdown = np.where( dvzd_dz[good_down[0][0:-1]] > 0.001 ) # Condition for when slope is close to zero
        small_erfdown = small_erfdown[0]
        
        delta_tau_down = np.zeros( (nz-1, nlam) ) # Define the delta-tau array for downflow
        # Here we are creating variables for the Taylor Expansion expr, so we don't have to rewrite them 100 times
        z1d = z_step[good_down[0][small_mdown]]   #z_step, left side of slab, downflow
        z2d = z_step[good_down[0][small_mdown]+1] #z_step, right side of slab, downflow
        mloc_down = dvzd_dz[good_down[0][small_mdown]]
        #Calculate delta-tau-down:
        if good_down[0][0:-1].size > 0:
            for l, v in enumerate(lamlist): #enumerate defines i, x => index, variable
                # Calculation for the ERF
                delta_tau_down[good_down[0][small_erfdown], l] = \
                    ind.kap*(0.5*(rhod_list[good_down[0][small_erfdown]]+rhod_list[good_down[0][small_erfdown]+1])) * (0.5 / dvzd_dz[good_down[0][small_erfdown]]) \
                        * np.abs( erf( (v - vzd_list[good_down[0][small_erfdown]])/ind.vth ) - erf( (v - vzd_list[good_down[0][small_erfdown]+1])/ind.vth )  )
                # Calculation for the Taylor Expansion
                gammad = (v - vzd_list[good_down[0][small_mdown]])/ind.vth #Define for simplicity
                delta_tau_down[good_down[0][small_mdown], l] = \
                    ind.kap*(0.5*(rhod_list[good_down[0][small_mdown]]+rhod_list[good_down[0][small_mdown]+1]))* \
                    (-1/(3*np.sqrt(np.pi)*ind.vth**3)) * ( np.exp(-gammad**2) * (z1d-z2d) * ( 3*ind.vth**2 + 3*mloc_down*ind.vth*(z2d-z1d)*gammad + \
                    mloc_down**2 * (z1d-z2d)**2 * (2*gammad**2 - 1) ) )

        # Plot One Plane
        #toto = ax.scatter( z_step[0:-1], [y]*(z_step.size-1), c=delta_tau_down[:,17], edgecolor='none', s=5, vmin=0,vmax=1.0, cmap="viridis"  )
        #toto = ax.scatter( [x]*(z_step.size-1), [y]*(z_step.size-1), c=delta_tau_down[:,17], edgecolor='none', s=5, vmin=0.0,vmax=1.0, cmap="Blues"  )
        # Plot One Ray
        #toto = ax.plot( z_step[0:-1], delta_tau_down[:,17]/(rhod_list[0:-1]))
        #toto = ax.plot(z_step[0:-1], delta_tau_down[:,17])

        # make a sum over the z, result in a nlam array.
        delta_tau = delta_tau_up + delta_tau_down
        #delta_tau = delta_tau_up
        #delta_tau = delta_tau_down
        
        #toto = ax.plot(z_step[0:-1], delta_tau[:,l])

        tau_list = np.zeros( (nz, nlam) )
        tau_list[0:-1,:] = np.cumsum( delta_tau[::-1,:], axis=0 )[::-1,:]

        #toto = ax.scatter( z_step, [y]*(z_step.size), c=tau_list[:,23], edgecolor='none', s=5, vmin=0,vmax=0.1, cmap='Greys'  )

        tau_back = tau_list[0,:]
        
	## Step 7: Calculate Absorption, Emission, Stokes V Contribution to Intensity
    
        ## For Ken, Stokes V Contribution to Intensity
        # Calculate Intensity along the ray from back (or star) to observer
        ##################################################
        # Use this block of code for one ray only
        #intensity = np.zeros(z_step.size)
        ##delta_iray = np.zeros(z_step.size)
        ##other_diray = np.zeros(z_step.size)
        #intensity_Bweight = np.zeros(z_step.size)
        #for zi in range(1,nz):
        #    #intensity[zi] = intensity[zi-1]*np.exp(-1*delta_tau[zi-1]) + srfc_list[zi]* (1.0 - np.exp(-1*delta_tau[zi-1]))
        #    intensity[zi] = intensity[zi-1]*np.exp(-1*delta_tau[zi-1,l]) + srfc_list[zi]* (1.0 - np.exp(-1*delta_tau[zi-1,l]))
        #    #delta_iray[zi] = intensity[zi] - intensity[zi-1]
        #    intensity_Bweight[zi] = (intensity[zi] - intensity[zi-1]) * Bz_Bstar_list[zi-1] * B_Bpole_list[zi-1]
        #    #other_diray[zi] = (srfc_list[zi] - intensity[zi-1]) * (1.0 - np.exp(-1*delta_tau[zi-1,l]))
        ##toto = ax.plot(z_step, intensity)
        ##toto = ax.plot(z_step, delta_iray)
        ##toto = ax.plot(z_step, other_diray)
        ##toto = ax.plot(z_step, intensity_Bweight)
        ##################################################
        
        # Update--Stokes V
        intensity = np.zeros( (nz, nlam) )
        intensity_Bweight = np.zeros( (nz, nlam) )
        for zi in range(1,nz):
            intensity[zi,:] = intensity[zi-1,:]*np.exp(-1*delta_tau[zi-1,:]) + (0.5*(srfc_list[zi]+srfc_list[zi-1]))* (1.0 - np.exp(-1*delta_tau[zi-1,:]))
            intensity_Bweight[zi,:] = (intensity[zi,:] - intensity[zi-1,:]) * Bz_Bstar_list[zi-1] * B_Bpole_list[zi-1]
                
        # iabs
        if x**2+y**2 <=1.0:
            iabs[i, j, :] = np.exp(-1*tau_back)*A
        
        # iemit, Update--Stokes V
        for l in range(0,nlam):
            iemit[i,j,l] =  A* np.sum(  0.5*(srfc_list[0:-1]+srfc_list[1:]) * np.exp(-1*tau_list[1:,l]) * (1.0 - np.exp(-1*delta_tau[:,l]) ) )
            intensity_antiV[i,j,l] = A * np.sum( intensity_Bweight[:,l] )
    
        #intensity = np.zeros(z_step.size)
        #print(intensity)
        #toto = ax.plot(z_step, intensity)
        #delta_iray = intensity[1:,:] - intensity[0:-1,:]
        #toto = ax.plot(z_step[1:], delta_iray)
        #intensity_Bweight = ( intensity[1:,:] - intensity[0:-1,:] ) * Bz_Bstar_list[0:-1] * B_Bpole_list[0:-1]
        #print(intensity_Bweight)
        #print(np.sum(intensity_Bweight))
        #toto = ax.plot(z_step, intensity_Bweight)

####################
#### END MATTER ####
####################

####
#Save Files
#### 

profile_abs = np.nansum( np.nansum(iabs, axis=1), axis=0)/xynorm
profile_emit = np.nansum( np.nansum(iemit, axis=1), axis=0)/xynorm
profile_antiV = np.nansum( np.nansum( intensity_antiV, axis=1), axis=0) / xynorm
#print(profile_antiV)

#pathlib.Path(ind.filepath).mkdir(exist_ok=False) 
#fileID = 'a'+str(np.degrees(ind.alp))+'_k'+str(ind.kap)
#copyfile('indat.py', ind.filepath+'/indat.py')

ascii.write(lamlist,filepath+'/wavelengths_vvinf.dat')

np.save(filepath+'/iabs_rawdata', iabs)
np.save(filepath+'/iemit_rawdata', iemit) #Use ind.filepath if need to import filepath from indat file
np.save(filepath+'/intensity_antiV_rawdata', intensity_antiV)
        
ascii.write(profile_abs,filepath+'/prof_abs.dat')
ascii.write(profile_emit,filepath+'/prof_emit.dat')
ascii.write(profile_antiV,filepath+'/prof_antiV.dat')

print('Output files saved.')

#### 
#Misc. “on the fly” plotting stuffs
####
 
#ax.plot( -1*lamlist, profile_abs, c='red' )
#ax.plot( -1*lamlist, profile_emit, c='blue' )
#ax.plot( -1*lamlist, profile_abs+profile_emit, c='k' )
#adu_v, adu_1, adu_2, adu_3 = np.genfromtxt('adu_real_up_out_0.00', unpack=True)
#ax.plot(-1*adu_v/2700, adu_1, ls='--', c='k')
#ax.plot(-1*adu_v/2700, adu_2, ls='--', c='red')
#ax.plot(-1*adu_v/2700, adu_3, ls='--', c='blue')

#plt.colorbar(toto)
#plt.show()

#print( 'This program ran in: {} seconds'.format((time.time()-start)))
print( 'This program ran in: {} minutes'.format( (time.time()-start)/60.0 ) )
print("Hooray! Time to make Plots!")
