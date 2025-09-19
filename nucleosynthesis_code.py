# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 21:32:52 2025

@author: Elina
"""
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
import time
print("Running script: Most recent update on 13/09/2025")

G = 6.6738e-11; c = 299792458.0; M_sun = 1.9885e30
CU_to_ms = 1000 * (M_sun * G / c**3) #time conversion from code units to ms
densconv = 1.619e10
km = 1.477 #solar masses to km
Tconv = 1.1604e1 #MeV to GK

#%% Load in data
model = "eoff"
EoS = "DD2"
levels = [5, 4, 3, 2, 1, 0]  #Refinement levels
n_j = 20                     #Number of parallel processes desired (to create interpolators)
outputfilepath = "results"   #Name of folder created to store nucleosynthesis data files

#%%
filename = ""                      #path to trajectory data
#%% Load in trajectory H5 file
data = []
with h5py.File(filename, 'r') as f:
    for particle_key in sorted(f.keys())[:]:  
        x = km*f[particle_key]['x_positions'][:]
        y = km*f[particle_key]['y_positions'][:]
        z = km*f[particle_key]['z_positions'][:]
        n_p = len(f.keys())
        print("length of x position array:", len(x))
        data.append((x, y, z))

print("loaded in data")
#%% Load in file paths
paths = [] #Input chronological list of simulation output files from each day of simulation

#%% Helper functions
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
            x, y, z = sim.gf.xyz["T"][it].get_level(l).coordinates_meshgrid()
        except:
            v_x = load_grid(f"T{l}_grid_{it}.pkl")
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



#%%
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
print("No. Iterations:", len(its))
print("No. Times:", len(times))

#%%Get coordinate bounds
if os.path.exists("coords_dict_nuc.pkl"):
    with open("coords_dict_nuc.pkl", "rb") as f:
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
    with open("coords_dict_nuc.pkl", "wb") as f:
         pickle.dump(coords_dict, f)

#%% Get RGIs. 
def get_rgi(path, it):
    sim = sd.SimDir(path)  # construct sim from path
    local_rgi_cache = {}

    for l in [5, 4, 3, 2, 1, 0]:
        try:
            T = sim.gf.xyz["igm_temperature"][it].get_level(l)
            rho = sim.gf.xyz["rho_b"][it].get_level(l)
            Y_e = sim.gf.xyz["igm_Ye"][it].get_level(l)
        except:
            T = load_grid(f"T{l}_grid_{it}.pkl")
            rho = load_grid(f"rho{l}_grid_{it}.pkl")
            Y_e = load_grid(f"Ye{l}_grid_{it}.pkl")

        shape = (2, 1, 0)
        T_data = np.transpose(T.data_xyz, shape)
        rho_data = np.transpose(rho.data_xyz, shape)
        Y_e_data = np.transpose(Y_e.data_xyz, shape)

        x_mesh, y_mesh, z_mesh = T.coordinates_meshgrid()
        x_1d = np.unique(x_mesh)
        y_1d = np.unique(y_mesh)
        z_1d = np.unique(z_mesh)

        local_rgi_cache[l] = {
            "T": RGI((x_1d, y_1d, z_1d), T_data, method='linear', bounds_error=False, fill_value=None),
            "rho": RGI((x_1d, y_1d, z_1d), rho_data, method='linear', bounds_error=False, fill_value=None),
            "Y_e": RGI((x_1d, y_1d, z_1d), Y_e_data, method='linear', bounds_error=False, fill_value=None),
        }

    return (path, it), local_rgi_cache



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

#%%
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

#%% Interpolates to find T, Y_e, rho along the path

def find_nuc_quantities(j, data, rgi_dict):
    xs = data[j][0]
    ys = data[j][1]
    zs = data[j][2]
    
    all_T = []
    all_Ye = []
    all_rho = []
    
    for i in range(len(its)):
        x, y, z = xs[i], ys[i], zs[i]
        point = (x, y, z)
        l = find_rl(i, x, y, z)
        #print("level, point", l, point)
        
        rgi_T = rgi_dict[its[i]][l]['T']
        rgi_Ye = rgi_dict[its[i]][l]['Y_e']
        rgi_rho = rgi_dict[its[i]][l]['rho']

        T = rgi_T(point)
        Y_e = rgi_Ye(point)
        rho = rgi_rho(point)
        print(T, Y_e, rho)
        
        all_T.append(T)
        all_Ye.append(Y_e)
        all_rho.append(rho)
        
    return f"particle_{j}", {
        "x": xs,
        "y": ys,
        "z": zs,
        "T": all_T,
        "Ye": all_Ye,
        "rho": all_rho,
        "times": times
    }
        
nuc_results = Parallel(n_jobs=1, backend="threading", verbose=500)(
    delayed(find_nuc_quantities)(j, data, rgi_dict)
    for j in range(n_p)
)

all_data = dict(nuc_results)
WriteScalarHDF5("nuc_results.h5", all_data)

#%%
output_dir = outputfilepath
os.makedirs(output_dir, exist_ok=True)
print(f"Directory created or exists: {os.path.abspath(output_dir)}")

for j in range(n_p):
    particle = all_data[f"particle_{j}"]
    t = np.array(particle["times"])*CU_to_ms 
    xs = np.array(particle["x"])*km
    ys = np.array(particle["y"])*km
    zs = np.array(particle["z"])*km
    T = np.log10(Tconv*np.array(particle["T"]))
    rho = np.log10(densconv*np.array(particle["rho"]))
    Y_e = np.array(particle["Ye"])

    # Ensure all arrays have same length
    n_steps = len(t)

    txt_filename = os.path.join(output_dir, f"particle_{j}_trajectory.txt")
    with open(txt_filename, "w") as f:
        f.write("t [ms]       x[km]       y[km]      z[km]      log(rho[cgs]) log(T[K])   Ye\n")
        f.write("----------------------------------------------------------------------------------\n")

        for i in range(n_steps):
            f.write("{:11.4E} {:11.4E} {:11.4E} {:11.4E} {:13.4E} {:11.4E} {:11.4E}\n".format(
                t[i], xs[i], ys[i], zs[i], rho[i], T[i], Y_e[i]
            ))



