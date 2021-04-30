## C. Erba
## Code to produce Velocity Maps from Asif's 3D MHD Sims
## In original form: 11/6/20

#########################
#### IMPORT COMMANDS ####
#########################
#import os, sys, inspect, pathlib, scipy, time

import numpy as np
import time
from astropy.io import ascii
from matplotlib.colors import to_rgba_array, LinearSegmentedColormap
import matplotlib.pyplot as plt

print('Program START')
start = time.time() # START THE CLOCK

##############################
##### COLORMAP FUNCTIONS #####
##############################

def discretemap(colormap, hexclrs):
    """
    Produce a colormap from a list of discrete colors without interpolation.
    """
    clrs = to_rgba_array(hexclrs)
    clrs = np.vstack([clrs[0], clrs, clrs[-1]])
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [ (i/(len(clrs)-2.), clrs[i, ki], clrs[i+1, ki]) for i in range(len(clrs)-1) ]
    return LinearSegmentedColormap(colormap, cdict)


class TOLcmaps(object):
    """
    Class TOLcmaps definition.
    """
    def __init__(self):
        """
        """
        self.cmap = None
        self.cname = None
        self.namelist = ('sunset_discrete', 'sunset', 'vibrant', 'YlOrBr')

        self.funcdict = dict(
            zip(self.namelist,
                (self.__sunset_discrete, self.__sunset)))

    def __sunset_discrete(self):
        """
        Define colormap 'sunset_discrete'.
        """
        '''
        clrs = ['#95211B','#B8221E','#DA2222','#E67932','#4EB265','#33BBEE','#0077BB','#332288'] #GOOD
        clrs = ['#95211B','#B8221E','#DA2222','#E67932','#E49C39','#8CBC68','#4EB265','#33BBEE','#0077BB','#332288'] #GOOD

        clrs = ['#95211B','#95211B','#B8221E','#B8221E','#DA2222','#DA2222','#E67932','#E67932', #Red
                '#E49C39','#E49C39','#DDAA3C','#A6BE54','#8CBC68','#8CBC68',
                '#4EB265','#4EB265','#33BBEE','#33BBEE','#0077BB','#0077BB','#332288','#332288'] #Blue #GOOD: Fine spacing colormap
        '''
        clrs = ['#332288','#332288','#2166ac','#2166ac','#4393c3','#4393c3','#92c5de','#92c5de',
               '#009988','#4eb265','#f7cb45','#ee8026',
               '#e65518','#e65518','#dc050c','#dc050c','#a5170e','#a5170e','#72190e','#72190e'][::-1]
        
        self.cmap = discretemap(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')

    def __sunset(self):
        """
        Define colormap 'sunset'.
        """
        clrs = ['#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF',
                '#EAECCC', '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D',
                '#A50026']
        self.cmap = LinearSegmentedColormap.from_list(self.cname, clrs)
        self.cmap.set_bad('#FFFFFF')
        
    def get(self, cname='sunset'):
        """
        Return requested colormap, default is 'sunset'.
        """
        self.cname = cname
        if cname == 'sunset':
            self.__sunset
        else:
            self.funcdict[cname]()
        return self.cmap

def tol_cmap(colormap=None):
    """
    Continuous and discrete color sets for ordered data.
    
    Return a matplotlib colormap.
    Parameter lut is ignored for all colormaps except 'rainbow_discrete'.
    """
    obj = TOLcmaps()
    return obj.get(colormap)
    
############################
##### ADD DIPOLE LOOPS #####
############################

def rotate_dipole(orig_coords, rot_ang):
    rot2D = np.array(((np.sin(rot_ang),-np.cos(rot_ang)),
                      (np.cos(rot_ang), np.sin(rot_ang))))
    diploop_rot=[]
    for i in range(0,len(orig_coords)):
        diploop_rot.append(np.matmul(rot2D,orig_coords[i]).tolist())
    return diploop_rot

rm_here = 2.3

diploop_rm=np.genfromtxt('diploop_rm_2p3.dat', skip_header=1, unpack=True)
diploop_neg_rm=np.genfromtxt('diploop_neg_rm_2p3.dat', skip_header=1, unpack=True)

##### FOR POLE #####
#plot_rm = rotate_dipole(diploop_rm,np.radians(0))
#plot_nrm = rotate_dipole(diploop_neg_rm,np.pi+np.radians(0))

##### FOR EQUATOR #####
plot_rm = rotate_dipole(diploop_rm,np.radians(90))
plot_nrm = rotate_dipole(diploop_neg_rm,np.pi+np.radians(90))

########################
#### IMPORT MHD SIM ####
########################

# Import datacube, assign variables

# cube200
# right now, we're reading in 1Ms cube
# 1Ms cube = 'cube200_T1oc_1Ms.dat'
# 875ks cube = 'cube200_T1oc_875ks.dat'
# 750ks cube = 'cube200_T1oc_750ks.dat'

nx=200
ny=200
nz=200

inx, iny, inz, invx, invy, invz, inbx, inby, inbz, inrho, inT = np.genfromtxt('cube200_T1oc_1Ms.dat', skip_header=17, unpack='true')

print('Datacube successfully read in')

# Various useful constants
Rstar = 8 * 7.e10 #cm
Vinf = 3200 # km/s
Bpole = 1100 #G
Mdot = 3.3e-7 # Msun/yr
rho_wind = Mdot * 2e33 / 3e7 / (4*np.pi*Vinf*1e5*Rstar**2) # in g/cm3
vth = 0.01 #vinf
kap = 1.0

#This creates data cubes, which are encoded in (x, y, z):
# x,y,z have been translated to Rstar units
# velocities have been translated into vinf units

cube_x = np.reshape(inx, (nx,ny,nz), order='F')/Rstar
cube_y = np.reshape(iny, (nx,ny,nz), order='F')/Rstar
cube_z = np.reshape(inz, (nx,ny,nz), order='F')/Rstar
cube_vz = np.reshape(invz, (nx,ny,nz), order='F')/Vinf/1e5

cube_vx = np.reshape(invx, (nx,ny,nz), order='F')/Vinf/1e5
cube_vy = np.reshape(invy, (nx,ny,nz), order='F')/Vinf/1e5
cube_vecv = ( cube_vx**2 + cube_vy**2 + cube_vz**2 )**0.5 # magnitude of vec{v}: |vec{v}|

#cube_bx = np.reshape(inbx, (nx,ny,nz), order='F')/Bpole
#cube_by = np.reshape(inby, (nx,ny,nz), order='F')/Bpole
#cube_bz = np.reshape(inbz, (nx,ny,nz), order='F')/Bpole  #bz/pole

#cube_rho = np.reshape(inrho, (nx,ny,nz), order='F')/rho_wind
#cube_r = ( cube_x**2 + cube_y**2 + cube_z**2 )**0.5

#########################
#### ACTUAL PLOTTING ####
#########################
figsize=(10,10)
fig, ax = plt.subplots(1, 1, figsize=figsize) #row, column
plt.rcParams.update({'font.size': 14,'font.family':'Times new roman'})

#### STAR ####
# Block out stuff behind star
box = plt.Rectangle( (-5.0,-1.0), 5.0, 2.0, color='lightgrey', zorder=2 )
ax.add_patch(box)
# Make and plot circle for star
circ = plt.Circle((0, 0), radius=1.0, linestyle='-', lw=1, color='k', fill=True, zorder=3)
ax.add_patch(circ)

#### Dipole Loops ####
# Add last closed dipole loop
ax.plot(np.asarray(plot_rm).T.tolist()[0],np.asarray(plot_rm).T.tolist()[1], color='dimgray', lw=3.0, ls='--', zorder=4)
ax.plot(np.asarray(plot_nrm).T.tolist()[0],np.asarray(plot_nrm).T.tolist()[1], color='dimgray', lw=3.0, ls='--', zorder=4)

#### Vmap ####

##### FOR POLE #####
#print(cube_x[100,0,0]) #This is the closest slice to x=0. It's x=0.024991026785714285
#im=ax.imshow(cube_vz[100,:,:], extent=[-5, 5, -5, 5], vmin=-1.0, vmax=1.0, cmap = tol_cmap('sunset_discrete')) #cmap = 'hsv' #choose x=0 slice
#im=ax.imshow(cube_vecv[100,:,:], extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap = 'YlOrRd') #cmap = 'hsv' #choose x=0 slice
#im=ax.imshow(np.log10(cube_rho[100,:,:]), extent=[-5, 5, -5, 5], cmap = 'plasma_r') #vmin=0.0, vmax=1.0, cmap = 'YlOrRd') #cmap = 'hsv' #choose x=0 slice

##### FOR EQUATOR #####
#print(cube_x[100,0,0]) #This is the closest slice to x=0. It's x=0.024991026785714285
im=ax.imshow(cube_vy[100,:,:].T, extent=[-5, 5, -5, 5], vmin=-1.0, vmax=1.0, cmap = tol_cmap('sunset_discrete')) #cmap = 'hsv' #choose x=0 slice

#
cb=fig.colorbar(im, shrink=0.85, aspect=10, pad=0.025, ticks=np.linspace(-1,1,11))
#cb=fig.colorbar(im, shrink=0.85, aspect=10, pad=0.025, ticks=np.linspace(0,1,11))
#cb=fig.colorbar(im, shrink=0.85, aspect=10, pad=0.025)
cb.ax.set_title(r'$v/v_{\infty}$')
#cb.ax.set_title(r'$Log(\rho/\rho_{w\ast})$')
#
ax.set_xticks(np.linspace(-5,5,11))
ax.set_yticks(np.linspace(-5,5,11))
ax.set_xticklabels(np.linspace(-5,5,11))
ax.set_yticklabels(np.linspace(-5,5,11))
#
ax.set_xlabel(r'Z (R/$R_{\ast}$)')
ax.set_ylabel(r'Y (R/$R_{\ast}$)')
#
ax.set_title(r'$V_z$')
#ax.set_title(r'| $\vec{V}$ |')
#ax.set_title(r'$Log(\rho/\rho_{w\ast})$')
ax.text(4., -5.9, 'To Observer >>>') #Observer Annotation
#
plt.show()
####################
#### Save Files ####
####################

# Savefig Commands
#fig.savefig('../../../figs/theta1OriC_vzmap_pole.jpg', bbox_inches='tight') #dpi=300,
#fig.savefig('../../../figs/theta1OriC_vecvmap_pole.jpg', bbox_inches='tight') #dpi=300,
#fig.savefig('../../../figs/theta1OriC_rhomap_pole.jpg', bbox_inches='tight') #dpi=300,
#fig.savefig('../../../figs/theta1OriC_vzmap_eq.jpg', bbox_inches='tight') #dpi=300,

print('Plots saved.')

#ascii.write(<array>,'<filename>.dat')
#print('Output files saved.')

print( 'This program ran in: {} minutes'.format( (time.time()-start)/60.0 ) )

