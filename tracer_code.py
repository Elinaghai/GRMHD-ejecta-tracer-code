# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:48:15 2025

@author: Elina
"""

print('script started')
print("RUNNING: latest version from 2025-08-04")

import scipy.interpolate as spint
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import threading
from multiprocessing import shared_memory
import os
import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm
from kuibit import simdir as sd
from collections import defaultdict


# Constants
G = 6.6738e-11; c = 299792458.0; M_sun = 1.9885e30
CU_to_ms = 1000 * (M_sun * G / c**3)
km = 1.477 #conversion from M to km
densconv = 1.619e10

#%% Load in data
model = "eoff"
case_name = "eoff/"
EoS = "DD2"
levels = [5, 4, 3, 2, 1, 0]  #Refinement levels
n_j = 25                    #Number of parallel processes desired (to create interpolators)

#%% Load in file paths
paths = [] #Input a chronological list of simulation output file paths

print("Number of file paths:", len(paths))
#%% Extract the iterations (and times corresponding to each) from the data
sim = sd.SimDir(paths[-1]) #25_04_24_115604
print("loaded in data")

its = sim.gf.xyz["vx"].available_iterations  
dt = its[1] - its[0]
print('Iteration step is:', dt)
times = sim.gf.xyz["vx"].times
lastit, firstit = its[-1], its[0]
lasttime = times[-1]          
print("Last iteration:", lastit)          #Some useful information
print("Last time (Code Units):", lasttime)
 
#%% Generate particles in rl=0 grid at the last iteration (across entire simulation box)
v_x=sim.gf.xyz["vx"][lastit].get_level(0)
x, y, z = v_x.coordinates_meshgrid()
n_p=10000000 #number of particles to generate 
N= 100000   #number of particles we want to select
 #Scatter plot
rand_x = np.random.uniform(x.min(), x.max(), n_p)
rand_y = np.random.uniform(y.min(), y.max(), n_p)
rand_z = np.random.uniform(z.min(), z.max(), n_p)

#%% Identify boundaries of each refinement level's grid
o_coords = {}             #Coordinates of the grid origin
uc_coords = {}            #Coordinates of the grid upper RH corner
for l in levels:
    vx = sim.gf.xyz['vx'][lastit].get_level(l)
    x, y, z = vx.coordinates_meshgrid()
    o_coords[l] = (x.min(), y.min(), z.min())
    uc_coords[l] = (x.max(), y.max(), z.max())
    print(f"origin of rl={l}", o_coords[l][0], o_coords[l][1], o_coords[l][2])
    print(f"upper corner of rl={l}", uc_coords[l][0], uc_coords[l][1], uc_coords[l][2])

#%% Constructs Regular Grid Interpolators (RGIs) for each refinement level at the last iteration and stores them in a dictionary.
def make_rgis(l):
    #get data
    #transpose
    shape = (2, 1, 0)
    crit = sim.gf.xyz["minus_one_minus_u_0"][lastit].get_level(l)        #-1-u_0 is a critical value which, when positive for a given particle, means that it is unbound to the system. 
    x, y, z = crit.coordinates_meshgrid()                                # v_x, v_y and v_z are coordinate velocities, with which a radial velocity can be calculated.
    
    critdata = crit.data_xyz
    vxdata = sim.gf.xyz["vx"][lastit].get_level(l).data_xyz              #data_xyz transposes the data, assuming that is originally aligned with (z, y, x) coordinates and must be changed in order to match a (x, y, z) coordinate format
    vydata = sim.gf.xyz["vy"][lastit].get_level(l).data_xyz              #In many cases though, this data_xyz transpose is unwanted, and here it is reversed. 
    vzdata = sim.gf.xyz["vz"][lastit].get_level(l).data_xyz              #If data is indeed stored matching (z, y, x) coordinates, shape = (0,0,0) can be used instead.
    rhodata = sim.gf.xyz["rho_b"][lastit].get_level(l).data_xyz          

    critdata = np.transpose(critdata, shape)
    vxdata = np.transpose(vxdata, shape)
    vydata = np.transpose(vydata, shape)
    vzdata = np.transpose(vzdata, shape)
    rhodata = np.transpose(rhodata, shape)
    
    x = np.unique(x)
    y = np.unique(y)
    z = np.unique(z)                                                     #Higher-order interpolation methods "cubic" and "quintic" can be chosen to replace the "linear" method. However, these only marginally improve accuracy in most cases.
    
    rgis = {
        "vx": RGI(points=(x, y, z), values=vxdata, method='linear', bounds_error=False, fill_value=0),        
        "vy": RGI(points=(x, y, z), values=vydata, method='linear', bounds_error=False, fill_value=0),        
        "vz": RGI(points=(x, y, z), values=vzdata, method='linear', bounds_error=False, fill_value=0),
        "rho": RGI(points=(x, y, z), values=rhodata, method='linear', bounds_error=False, fill_value=0),
        "crit": RGI(points=(x, y, z), values=critdata, method='linear', bounds_error=False, fill_value=0)
    }
    
    return l, rgis

all_rgis = Parallel(n_jobs=6, backend="loky")(delayed(make_rgis)(l) for l in levels)
rgis = {level: interp_dict for level, interp_dict in all_rgis}
#%%
def select_particles(i):
    point = (rand_x[i], rand_y[i], rand_z[i])

    if (o_coords[5][0] <= rand_x[i] <= uc_coords[5][0] and o_coords[5][1] <= rand_y[i] <= uc_coords[5][1] and o_coords[5][2] <= rand_z[i] <= uc_coords[5][2]):  
         l=5
    elif (o_coords[4][0] <= rand_x[i] <= uc_coords[4][0] and o_coords[4][1] <= rand_y[i] <= uc_coords[4][1] and o_coords[4][2] <= rand_z[i] <= uc_coords[4][2]):
         l=4
    elif (o_coords[3][0] <= rand_x[i] <= uc_coords[3][0] and o_coords[3][1] <= rand_y[i] <= uc_coords[3][1] and o_coords[3][2] <= rand_z[i] <= uc_coords[3][2]):  
         l=3
    elif (o_coords[2][0] <= rand_x[i] <= uc_coords[2][0] and o_coords[2][1] <= rand_y[i] <= uc_coords[2][1] and o_coords[2][2] <= rand_z[i] <= uc_coords[2][2]):
         l=2
    elif (o_coords[1][0] <= rand_x[i] <= uc_coords[1][0] and o_coords[1][1] <= rand_y[i] <= uc_coords[1][1] and o_coords[1][2] <= rand_z[i] <= uc_coords[1][2]):
         l=1
    else:  
         l=0
         
    rgi_crit = rgis[l]['crit']                                #We check if a particle is enclosed by the smallest, most refined grid first before defaulting to larger, coarser grids.
    rgi_rho = rgis[l]['rho']                                  #This way, we identify the finest grid that contains a given particle and interpolate for quantities using said grid, thus minimising interpolation errors.
    rgi_vx = rgis[l]['vx']                        
    rgi_vy = rgis[l]['vy']
    rgi_vz = rgis[l]['vz']
    
    
    crit = rgi_crit(point)
    vx = rgi_vx(point)
    vy = rgi_vy(point)
    vz = rgi_vz(point)
    rho = rgi_rho(point)
    
    print(crit, rho, vx, vy, vz)
    
    r = np.sqrt(rand_x[i]**2 + rand_y[i]**2 + rand_z[i]**2)  
    v_rad = (rand_x[i]*vx + rand_y[i]*vy + rand_z[i]*vz)/r 

    if (crit > 0) and (v_rad > 0):                           #This selects particles which are a) unbound to the system, and b) radially moving outwards. 
        return (                                             #These constitute our criteria for ejected particles.
            rand_x[i], rand_y[i], rand_z[i],
            vx, vy, vz,
            r,
            rho
        )
    else:
        return None

results = Parallel(n_jobs=200, backend="loky", verbose=10)(
    delayed(select_particles)(i) for i in range(n_p)
)

filtered = [r for r in results if r is not None]

if filtered:

    arr = np.array(filtered)                                                #Particles which satisfy criteria are saved to arrays.
    ejecta_x, ejecta_y, ejecta_z = arr[:, 0], arr[:, 1], arr[:, 2]
    ejecta_vx, ejecta_vy, ejecta_vz = arr[:, 3], arr[:, 4], arr[:, 5]
    ejecta_r, rho_ejecta = arr[:, 6], arr[:, 7]
else:
    # Create empty arrays with float dtype
    ejecta_x = ejecta_y = ejecta_z = np.array([])
    ejecta_vx = ejecta_vy = ejecta_vz = np.array([])
    ejecta_r = rho_ejecta = np.array([])

#%% EJECTA PLOTS
print('Making ejecta plots')
os.makedirs("plots_3D", exist_ok=True)
#3D PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter3d = ax.scatter(km*ejecta_x, km*ejecta_y, km*ejecta_z, c=densconv*rho_ejecta, cmap='viridis')
ax.set_xlabel('X / km')
ax.set_ylabel('Y / km')
ax.set_zlabel('Z / km')
cbar = plt.colorbar(scatter3d, ax=ax, shrink=0.75, aspect=10, pad=0.2)
cbar.set_label('Interpolated Baryonic Density / g cm^-3')  
plt.savefig("plots_3D/distribution_3d.png", dpi=300, bbox_inches='tight')
plt.close()

#SLICES
fig = plt.figure()
scatterxz = plt.scatter(km*ejecta_x, km*ejecta_z, c=densconv*rho_ejecta, cmap='viridis')
plt.xlabel('X / km')
plt.ylabel('Z / km')
cbar = plt.colorbar(scatterxz, shrink=0.75, aspect=10, pad=0.2)
cbar.set_label('Interpolated Baryonic Density / g cm^-3')  
plt.savefig("plots_3D/distribution_XZplane.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

fig = plt.figure()
scatterxy = plt.scatter(km*ejecta_x, km*ejecta_y, c=densconv*rho_ejecta, cmap='viridis')
plt.xlabel('X / km')
plt.ylabel('Y / km')
cbar = plt.colorbar(scatterxy, shrink=0.75, aspect=10, pad=0.2)
cbar.set_label('Interpolated Baryonic Density / g cm^-3')  
plt.savefig("plots_3D/distribution_XYplane.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

fig = plt.figure()
scatteryz = plt.scatter(km*ejecta_y, km*ejecta_z, c=densconv*rho_ejecta, cmap='viridis')
plt.xlabel('Y / km')
plt.ylabel('Z / km')
cbar = plt.colorbar(scatteryz, shrink=0.75, aspect=10, pad=0.2)
cbar.set_label('Interpolated Baryonic Density / g cm^-3')  
plt.savefig("plots_3D/distribution_YZplane.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

del(fig)

#%% Particle selection using density and radial weightings.
w = ejecta_r*rho_ejecta
w_sum = np.sum(w)
w_norm = w/w_sum

print("Sum of weights (should be 1):", np.sum(w_norm))                         #should be 1
print("Total no. ejecta particles", len(ejecta_x))
try:
   index_p = np.random.choice(len(ejecta_x), size=N, replace=False, p=w_norm)
except:
   index_p = np.random.choice(len(ejecta_x), size=len(ejecta_x)-1, replace=False, p=w_norm)
   print("More points requested than generated. Using n particles, where n=", len(ejecta_x)-1)
   
selected_x = ejecta_x[index_p]
selected_y = ejecta_y[index_p]
selected_z = ejecta_z[index_p]

selected_vx = ejecta_vx[index_p]
selected_vy = ejecta_vy[index_p]
selected_vz = ejecta_vz[index_p]

plt.figure()
plt.title("Particles in the ejecta generated by weighting densities")
scatterxy1 = plt.scatter(km*selected_x, km*selected_y, c=densconv*rho_ejecta[index_p], s=5)
plt.xlabel("Distance / km")
plt.ylabel("Distance / km")
cbar = plt.colorbar(scatterxy1, shrink=0.75, aspect=10, pad=0.2)
cbar.set_label('Interpolated Baryonic Density / g cm^-3')  
plt.show()
plt.savefig("plots_3D/distribution_XYplane_selected.png", dpi=300, bbox_inches='tight')
plt.close()
#%% Some helpful functions.

def WriteDictionaryVar(h5f, data):  
     for varname in sorted(list(data)):  
         x_data = data[varname]  
         x_type = type(x_data)  
         if x_type == dict:  
             grp = h5f.create_group(varname)  
             WriteDictionaryVar(grp, x_data)  
         else:  
             if x_type != np.ndarray:  
                 values = np.array(x_data)  
             else:  
                 values = x_data  
             try:  
                 h5f.create_dataset(varname, data=values)  
             except:  
                 print("Error creating dataset ", varname)  
   
def WriteScalarHDF5(file, data, group='', mode='a'):  
     f = h5py.File(file, mode)  
     if len(group) > 0:  
         try:  
             grp = f.create_group(group)  
         except:  
             grp = f[group]  
         grp = f[group]  
     else:  
         grp = f  
     WriteDictionaryVar(grp, data)  
     f.close()  
     return 1   
 
def save_grids(grids, filename):
    with open(filename, "wb") as f:
        pickle.dump(grids, f)

def load_grid(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def get_coord_bounds(path, it, time):         #Coordinate boundaries must be computed at each iteration (especially where moving grids are utilised)
    sim = sd.SimDir(path)                                                      
    coord_bounds = {
        "min": {},
        "max": {},
        "time": float(time)
    }

    for l in [5, 4, 3, 2, 1]:
        try:
            x, y, z = sim.gf.xyz["vx"][it].get_level(l).coordinates_meshgrid()
        except:
            v_x = load_grid(f"vx{l}_grid_{it}.pkl")
            x, y, z = v_x.coordinates_meshgrid()

        coord_bounds["min"][l] = {
            'x': float(np.min(x)),
            'y': float(np.min(y)),
            'z': float(np.min(z)),
        }
        coord_bounds["max"][l] = {
            'x': float(np.max(x)),
            'y': float(np.max(y)),
            'z': float(np.max(z)),
        }

    return it, coord_bounds

#%%Get its and times from dictionaries - for tracing. Sorted with no duplicates.
with open("its_dict.pkl", "rb") as f:  
    its_dict = pickle.load(f)
    
all_tasks = []
for path, data in its_dict.items():
    its = data["iterations"]
    times = data["times"]
    all_tasks.extend(zip([path] * len(its), its, times))

# check a sample
print(f"Loaded {len(all_tasks)} (path, it, time) entries")
print(all_tasks[:20])  # Show first 20

its = sorted(set(
    it for d in its_dict.values() for it in d["iterations"]
))
times = sorted(set(
    t for d in its_dict.values() for t in d["times"]
))

print("iterations", its)
print("times", times)

#%%Get coordinate bounds
if os.path.exists("coords_dict.pkl"):
    with open("coords_dict.pkl", "rb") as f:
         coords_dict = pickle.load(f)
         print("Loaded coordinate boundary dictionary from pickle file")
    
else:
    results = Parallel(
        n_jobs=n_j,
        backend="loky",
        verbose=500
    )(
        delayed(get_coord_bounds)(path, it, time)
        for path, it, time in all_tasks
    )
    print("Making coordinate boundary dictionary from simulation data.")
 
    coords_dict = dict(results)
    with open("coords_dict.pkl", "wb") as f:
         pickle.dump(coords_dict, f)

#%%
def get_rgi(path, it):
    sim = sd.SimDir(path)  
    local_rgi_cache = {}

    for l in [5, 4, 3, 2, 1, 0]:
        try:
            v_x = sim.gf.xyz["vx"][it].get_level(l)
            v_y = sim.gf.xyz["vy"][it].get_level(l)
            v_z = sim.gf.xyz["vz"][it].get_level(l)
        except:
            v_x = load_grid(f"vx{l}_grid_{it}.pkl")
            v_y = load_grid(f"vy{l}_grid_{it}.pkl")
            v_z = load_grid(f"vz{l}_grid_{it}.pkl")

        shape = (2, 1, 0)
        vx_data = np.transpose(v_x.data_xyz, shape)
        vy_data = np.transpose(v_y.data_xyz, shape)
        vz_data = np.transpose(v_z.data_xyz, shape)

        x_mesh, y_mesh, z_mesh = v_x.coordinates_meshgrid()
        x_1d = np.unique(x_mesh)
        y_1d = np.unique(y_mesh)
        z_1d = np.unique(z_mesh)

        local_rgi_cache[l] = {
            "vx": RGI((x_1d, y_1d, z_1d), vx_data, method='linear', bounds_error=False, fill_value=None),
            "vy": RGI((x_1d, y_1d, z_1d), vy_data, method='linear', bounds_error=False, fill_value=None),
            "vz": RGI((x_1d, y_1d, z_1d), vz_data, method='linear', bounds_error=False, fill_value=None),
        }

    return (path, it), local_rgi_cache


# Flatten (path, it) pairs
seen = set()
task_list = []
for path, data in its_dict.items():
    for it in data["iterations"]:
        key = (path, it)
        if key not in seen:
            task_list.append(key)
            seen.add(key)
print("First 10 elements of list of tasks:", task_list[:10])

# Run in parallel
rgi_dict_items = Parallel(n_jobs=100, backend="loky", verbose=5)(
    delayed(get_rgi)(path, it) for path, it in task_list
)

# Assemble into a dictionary: keys are (path, it)
rgi_dict = dict(rgi_dict_items)
rgi_dict = {
    it: val
    for (path, it), val in rgi_dict.items()
}

#%% FIND RL
def find_rl(i, x_pos, y_pos, z_pos):
    mincoords = coords_dict[its[i]]['min']
    maxcoords = coords_dict[its[i]]['max']

    for lvl in [5, 4, 3, 2, 1]:
        xi, xf = mincoords[lvl]['x'], maxcoords[lvl]['x']
        yi, yf = mincoords[lvl]['y'], maxcoords[lvl]['y']
        zi, zf = mincoords[lvl]['z'], maxcoords[lvl]['z']

        if (xi <= x_pos <= xf and yi <= y_pos <= yf and zi <= z_pos <= zf):
            return lvl

    return 0

#%% RK6 backwards tracer function
print("starting tracer")

def trace_particles(j, rgi_dict, its, times, selected_x, selected_y, selected_z, selected_vx, selected_vy, selected_vz):
    random_x = selected_x[j]
    random_y = selected_y[j]
    random_z = selected_z[j]
    
    random_vx = selected_vx[j]
    random_vy = selected_vy[j]
    random_vz = selected_vz[j]
    
    print(random_x, "random x")  
    print(random_y, "random y")  
    print(random_z, "random z")
    print(f"processing particle number {j+1}")
    
    vx_interp = np.zeros_like(times)  
    vy_interp = np.zeros_like(times)  
    vz_interp = np.zeros_like(times)
    x_positions = np.zeros_like(times)  
    y_positions = np.zeros_like(times)  
    z_positions = np.zeros_like(times)
    
    vx_interp[-1] = random_vx  
    vy_interp[-1] = random_vy  
    vz_interp[-1] = random_vz
    
    x_positions[-1] = random_x  
    y_positions[-1] = random_y  
    z_positions[-1] = random_z
    
    for i in range(len(times) - 1, 0, -1): 
        
        dt = times[i] - times[i-1]  
        
        it = its[i-1]
        
        l = find_rl(i-1, x_positions[i], y_positions[i], z_positions[i])
        rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][l]['vx'], rgi_dict[it][l]['vy'], rgi_dict[it][l]['vz']

        # k1
        k1x = rgi_vx((x_positions[i], y_positions[i], z_positions[i]))
        k1y = rgi_vy((x_positions[i], y_positions[i], z_positions[i]))
        k1z = rgi_vz((x_positions[i], y_positions[i], z_positions[i]))

        # k2
        L = find_rl(i-1, x_positions[i] - (1/3)*dt*k1x, y_positions[i] - (1/3)*dt*k1y, z_positions[i] - (1/3)*dt*k1z)
        if l != L:
            rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L]['vx'], rgi_dict[it][L]['vy'], rgi_dict[it][L]['vz']

        k2x = rgi_vx((x_positions[i] - (1/3)*dt*k1x, y_positions[i] - (1/3)*dt*k1y, z_positions[i] - (1/3)*dt*k1z))
        k2y = rgi_vy((x_positions[i] - (1/3)*dt*k1x, y_positions[i] - (1/3)*dt*k1y, z_positions[i] - (1/3)*dt*k1z))
        k2z = rgi_vz((x_positions[i] - (1/3)*dt*k1x, y_positions[i] - (1/3)*dt*k1y, z_positions[i] - (1/3)*dt*k1z))

        # k3
        L2 = find_rl(i-1, x_positions[i] - (1/6)*dt*k1x - (1/6)*dt*k2x, y_positions[i] - (1/6)*dt*k1y - (1/6)*dt*k2y, z_positions[i] - (1/6)*dt*k1z - (1/6)*dt*k2z)
        if L != L2:
            rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L2]['vx'], rgi_dict[it][L2]['vy'], rgi_dict[it][L2]['vz']

        k3x = rgi_vx((x_positions[i] - (1/6)*dt*k1x - (1/6)*dt*k2x, y_positions[i] - (1/6)*dt*k1y - (1/6)*dt*k2y, z_positions[i] - (1/6)*dt*k1z - (1/6)*dt*k2z))
        k3y = rgi_vy((x_positions[i] - (1/6)*dt*k1x - (1/6)*dt*k2x, y_positions[i] - (1/6)*dt*k1y - (1/6)*dt*k2y, z_positions[i] - (1/6)*dt*k1z - (1/6)*dt*k2z))
        k3z = rgi_vz((x_positions[i] - (1/6)*dt*k1x - (1/6)*dt*k2x, y_positions[i] - (1/6)*dt*k1y - (1/6)*dt*k2y, z_positions[i] - (1/6)*dt*k1z - (1/6)*dt*k2z))

        # k4
        L3 = find_rl(i-1, x_positions[i] - (1/8)*dt*k1x - 0*dt*k2x - (3/8)*dt*k3x, y_positions[i] - (1/8)*dt*k1y - 0*dt*k2y - (3/8)*dt*k3y,
             z_positions[i] - (1/8)*dt*k1z - 0*dt*k2z - (3/8)*dt*k3z)
        if L2 != L3:
           rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L3]['vx'], rgi_dict[it][L3]['vy'], rgi_dict[it][L3]['vz']

        k4x = rgi_vx((x_positions[i] - (1/8)*dt*k1x - (3/8)*dt*k3x,
              y_positions[i] - (1/8)*dt*k1y - (3/8)*dt*k3y,
              z_positions[i] - (1/8)*dt*k1z - (3/8)*dt*k3z))
        k4y = rgi_vy((x_positions[i] - (1/8)*dt*k1x - (3/8)*dt*k3x,
              y_positions[i] - (1/8)*dt*k1y - (3/8)*dt*k3y,
              z_positions[i] - (1/8)*dt*k1z - (3/8)*dt*k3z))
        k4z = rgi_vz((x_positions[i] - (1/8)*dt*k1x - (3/8)*dt*k3x,
              y_positions[i] - (1/8)*dt*k1y - (3/8)*dt*k3y,
              z_positions[i] - (1/8)*dt*k1z - (3/8)*dt*k3z))

        # k5
        L4 = find_rl(i-1,
             x_positions[i] - (1/2)*dt*k1x + 0*dt*k2x + (3/2)*dt*k3x - 2*dt*k4x,
             y_positions[i] - (1/2)*dt*k1y + 0*dt*k2y + (3/2)*dt*k3y - 2*dt*k4y,
             z_positions[i] - (1/2)*dt*k1z + 0*dt*k2z + (3/2)*dt*k3z - 2*dt*k4z)
        if L3 != L4:
           rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L4]['vx'], rgi_dict[it][L4]['vy'], rgi_dict[it][L4]['vz']

        k5x = rgi_vx((x_positions[i] - (1/2)*dt*k1x + (3/2)*dt*k3x - 2*dt*k4x,
              y_positions[i] - (1/2)*dt*k1y + (3/2)*dt*k3y - 2*dt*k4y,
              z_positions[i] - (1/2)*dt*k1z + (3/2)*dt*k3z - 2*dt*k4z))
        k5y = rgi_vy((x_positions[i] - (1/2)*dt*k1x + (3/2)*dt*k3x - 2*dt*k4x,
              y_positions[i] - (1/2)*dt*k1y + (3/2)*dt*k3y - 2*dt*k4y,
              z_positions[i] - (1/2)*dt*k1z + (3/2)*dt*k3z - 2*dt*k4z))
        k5z = rgi_vz((x_positions[i] - (1/2)*dt*k1x + (3/2)*dt*k3x - 2*dt*k4x,
              y_positions[i] - (1/2)*dt*k1y + (3/2)*dt*k3y - 2*dt*k4y,
              z_positions[i] - (1/2)*dt*k1z + (3/2)*dt*k3z - 2*dt*k4z))

        # k6
        L5 = find_rl(i-1,
             x_positions[i] + (3/7)*dt*k1x - (8/7)*dt*k3x - (6/7)*dt*k4x + (12/7)*dt*k5x,
             y_positions[i] + (3/7)*dt*k1y - (8/7)*dt*k3y - (6/7)*dt*k4y + (12/7)*dt*k5y,
             z_positions[i] + (3/7)*dt*k1z - (8/7)*dt*k3z - (6/7)*dt*k4z + (12/7)*dt*k5z)
        if L4 != L5:
           rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L5]['vx'], rgi_dict[it][L5]['vy'], rgi_dict[it][L5]['vz']

        k6x = rgi_vx((x_positions[i] + (3/7)*dt*k1x - (8/7)*dt*k3x - (6/7)*dt*k4x + (12/7)*dt*k5x,
              y_positions[i] + (3/7)*dt*k1y - (8/7)*dt*k3y - (6/7)*dt*k4y + (12/7)*dt*k5y,
              z_positions[i] + (3/7)*dt*k1z - (8/7)*dt*k3z - (6/7)*dt*k4z + (12/7)*dt*k5z))
        k6y = rgi_vy((x_positions[i] + (3/7)*dt*k1x - (8/7)*dt*k3x - (6/7)*dt*k4x + (12/7)*dt*k5x,
              y_positions[i] + (3/7)*dt*k1y - (8/7)*dt*k3y - (6/7)*dt*k4y + (12/7)*dt*k5y,
              z_positions[i] + (3/7)*dt*k1z - (8/7)*dt*k3z - (6/7)*dt*k4z + (12/7)*dt*k5z))
        k6z = rgi_vz((x_positions[i] + (3/7)*dt*k1x - (8/7)*dt*k3x - (6/7)*dt*k4x + (12/7)*dt*k5x,
              y_positions[i] + (3/7)*dt*k1y - (8/7)*dt*k3y - (6/7)*dt*k4y + (12/7)*dt*k5y,
              z_positions[i] + (3/7)*dt*k1z - (8/7)*dt*k3z - (6/7)*dt*k4z + (12/7)*dt*k5z))

        # Final update
        x_positions[i-1] = x_positions[i] - dt*(7/90*k1x + 0*k2x + 32/90*k3x + 12/90*k4x + 32/90*k5x + 7/90*k6x)
        y_positions[i-1] = y_positions[i] - dt*(7/90*k1y + 0*k2y + 32/90*k3y + 12/90*k4y + 32/90*k5y + 7/90*k6y)
        z_positions[i-1] = z_positions[i] - dt*(7/90*k1z + 0*k2z + 32/90*k3z + 12/90*k4z + 32/90*k5z + 7/90*k6z)

        # Update interpolated velocities
        L_f = find_rl(i-1, x_positions[i-1], y_positions[i-1], z_positions[i-1])
        if L5 != L_f:
           rgi_vx, rgi_vy, rgi_vz = rgi_dict[it][L_f]['vx'], rgi_dict[it][L_f]['vy'], rgi_dict[it][L_f]['vz']

        vx_interp[i-1] = rgi_vx((x_positions[i-1], y_positions[i-1], z_positions[i-1]))
        vy_interp[i-1] = rgi_vy((x_positions[i-1], y_positions[i-1], z_positions[i-1]))
        vz_interp[i-1] = rgi_vz((x_positions[i-1], y_positions[i-1], z_positions[i-1]))

        print(x_positions[i-1], y_positions[i-1], z_positions[i-1])

    # Save data
    particle_key = f"particle_{j}"  

   
    return particle_key, {
    "x_positions": x_positions,
    "y_positions": y_positions,
    "z_positions": z_positions,
    "vx_interp": vx_interp,
    "vy_interp": vy_interp,
    "vz_interp": vz_interp,
    "times": times,
}


#%%
N = len(selected_x) #relabel N just in case fewer particles were detected than expected

results = Parallel(n_jobs=2, backend="threading", verbose=500)(
    delayed(trace_particles)(j, rgi_dict, its, times, selected_x, selected_y, selected_z, selected_vx, selected_vy, selected_vz)
    for j in range(N)
)

all_particle_data = dict(results)

 # Save all particle data to HDF5 file  
filename = f"particle_trajectories_{EoS}{model}_{N}.h5"  
WriteScalarHDF5(filename, all_particle_data, mode='w')  
print(f"Saved particle trajectories to {filename}") 
#%%

filtered_data = {
    key: data for key, data in all_particle_data.items()
    if np.sqrt(data["x_positions"][0]**2 + data["y_positions"][0]**2 + data["z_positions"][0]**2) < 100
}

    
N = len(filtered_data)
filtered_filename = f"filtered_trajectories_{EoS}{model}_{N}"
WriteScalarHDF5(filtered_filename, filtered_data, mode='w')  
print(f"Saved particle trajectories to {filtered_filename}") 
print("End of script.")
    
    
    
    
    
    


