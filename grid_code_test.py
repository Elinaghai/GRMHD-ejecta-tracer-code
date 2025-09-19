# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:33:31 2025

@author: Elina
"""

print('Grid script started')
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

#%% Load in paths, get iterations. 
paths = ["/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_24_150154",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_24_150653",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_25_152418",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_26_163002",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_27_172131",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_28_092538",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_29_092730",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_30_102754",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_03_31_111655",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_01_143631",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_02_182231",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_03_182425",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_04_182558",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_05_182800",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_06_182810",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_07_191638",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_08_191743",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_09_191908",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_11_144651",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_12_144724",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_13_144803",
    "/gpfs/scratch/uv106/venturif/prod_runs/hshen_eoff_mn5/data_checkpoint/25_04_24_115226"]
    
#%% Define function that saves our data
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
    
#%% Get spacings. EDITED FOR LAPTOP (sim)
sim = sd.SimDir(paths[-1])
#sim = sd.SimDir(r"C:\Users\Elina\Downloads\Master's project\data_times") 
its = sim.gf.xyz["vx"].available_iterations
lastit = its[-1]
n_j = 20
levels = [5, 4, 3, 2, 1, 0]

def get_spacings(L):
    dx = sim.gf.xyz["vx"][lastit].dx_at_level(L)
    return {f"{L}": dx}

spacing_dicts = Parallel(
    n_jobs=n_j,
    backend='loky',
    verbose=n_j
)(
    delayed(get_spacings)(L)
    for L in levels
)

spacings = {k: v for d in spacing_dicts for k, v in d.items()}

print("spacing at rl=5 at last frame", spacings['5'])
#%% CHANGED FOR LAPTOP: 
n_j = 2
vars_to_grid = {'vx':'vx','vy':'vy', 'vz':'vz', 'igm_temperature':'T', 'rho_b':'rho', 'igm_Ye':'Ye'}

def make_grids(path):
    messages = []
    working_paths = []
    working_its = []
    
    sim = sd.SimDir(path)
    try:
       simvx = sim.gf.xyz["vx"]                     #1. Can we open the data already with get_level?
       working_paths.append(path)
    except KeyError:
       msg = f"'vx' variable not present in simulation at {path} — skipping to next path"
       print(msg)
       messages.append(msg)
       
       return {
            "path": path,
            "iterations": [],
            "times": [],
            "log": messages,

        }
    print(f"Loaded simulation at {path}")
    
    its = sim.gf.xyz["vx"].available_iterations
    times = sim.gf.xyz["vx"].times
    print("iterations:", its)                             
    print("times", times)
    
    for it in its:
      for L in [5, 4, 3, 2, 1]:                            #RL=0 omitted as this is the coarsest level and is guaranteed to exist if data is not corrupted/absent
        try:                                              #List can be changed, but checking all levels (and not just 5 and 4) is more robust
            sim.gf.xyz['vx'][it].get_level(L)
            working_its.append(it)
            continue
        except Exception:
            try:
                v_x = sim.gf.xyz['vx'][it].get_level(L-1)
            except:
                try:
                   v_x = sim.gf.xyz['vx'][it].get_level(0)
                except:
                     messages.append(f"Iteration {it} in file {path} is probably corrupted/empty - skipping to next iteration")
                     continue
            try:
               x, y, z = v_x.coordinates_meshgrid()
               gridshape = x.shape

               dxx = tuple(float(v) for v in spacings[f'{L}'])
               origin = (dxx[0]*gridshape[0], dxx[1]*gridshape[1], dxx[2]*gridshape[2])
               gridshape = tuple(gridshape)

               for var, pref in vars_to_grid.items():
                   grd = sim.gf.xyz[var][it].to_UniformGridData(
                       shape=gridshape,
                       x0=origin,
                       dx=dxx,
                       resample=True
                   )
                   save_grids(grd, f"{pref}{L}_grid_{it}.pkl")
                   print("Created and saved grid.")
                   del grd
               messages.append(f"[{it}] grid saved")
               working_its.append(it)
               
            except Exception as e:
               messages.append(f"Failed to make fallback grid for it={it}: {e}")
               print(f"Failed to make fallback grid for it={it}: {e}")
               continue  # Continue to next iteration
    
    return {
        "path": path,
        "iterations": working_its,
        "times": times,
        "log": messages,

    }

results = Parallel(
    n_jobs=n_j,                  #CHANGE BACK TO 20
    backend='loky', 
    verbose=n_j
)(
    delayed(make_grids)(path) for path in paths
)
print ("grids made!")

#Make a dictionary of iterations to use later
its_dict = {
    res["path"]: {"iterations": res["iterations"], "times": res["times"]}
    for res in results
}

with open("its_dict.pkl", "wb") as f:
    pickle.dump(its_dict, f)
    
print("End of script.")
