# GRMHD-ejecta-tracer-code
@author: Elina Ghai, Imperial College London
Email: eg1022@ic.ac.uk (or elina.ghai@hotmail.co.uk after June 2026)
"""

'''
                                                   Motivations
.......................................................................................................................
When neutron stars collide and coalesce in a Binary Neutron Star (BNS) merger, a non-negligible fraction of
neutron-rich, hot and dense matter is ejected into space. Due to the extremely high abundance of neutrons
in these ejecta, and the rapid cooling/expansion that they undergo as they are expelled from the remnant, 
BNS ejecta provide an environment ripe for R-process nucleosynthesis.

Typical computational analysis of this phenomenon consists of computing so-called "tracers" within GRMHD simulations 
of BNS mergers. These are lagrangian particles which represent the motion of fluid elemnents within the ejecta, whose 
trajectories (and associated quantities relevant for nucleosynthesis) are used to compute the final abundances 
resulting from nucleosynthesis in the ejecta (utlising nucleosynthesis networks).

However, computing tracers within simulations is restrictive. This method only allows for a limited number of tracers
to be calculated, and does not a) allow for retrospective analysis of previous simulations, nor
 b) allow for multiple ejecta types to be considered. 
This last point is especially relevant.
Fundamentally, most GRMHD simulations cannot provide sufficient tracers to gain an appreciable picture of the ejecta.

This produces the need for a system of codes which generate, select and trace lagrangian particles outside 
the simulation itself- which is the result of this work. It can select all types of ejecta, and traces particles at 
a rate of ~150 particles per minute, per node, for a dataset of ~250 iterations. 

.......................................................................................................................
                                                     The codes
.......................................................................................................................
Here I present a post-processing framework consisting of 3 codes - grid_code.py, tracer_code.py and nucleo_tracer.py, 
designed to be executed in that order.

1. grid_code.py : (Linear complexity with no. simulation files)

This resamples simulation to grids in instances where there are multiple patches in a given refinement level's
grid box. This can occur in simulations utlising the "moving meshes" approach, where nested grids follow the motion
of each star and overlap during the merger. Generally, only the 2 finest refinement levels require resampling, 
but for generality, data across all refinement levels is tested.

2. tracer_code.py : (Linear complexity with no. particles + no. iterations)

tracer_code.py generates n_p random particles at the last frame of the simulation, and interpolates for quantities 
-1-u_0 and v_r, where v_r is the radial velocity of the particles (calculated from interpolated coordinate velocities).
We keep only particles for which -1-u_0>0, v_r>0, which are hence unbound and radially moving outwards.
These form part of the ejecta.

N particles are selected from the particles which are in the ejecta, based on a radial and density
weighting. If N is larger than the total number of particles generated in the ejecta, no weighted
selection will be applied.

Regular Grid Interpolators (RGIs) are constructed for each refinement level at each iteration and are
stored in a dictionary. These RGIs are used to interpolate velocity values which calculate backwards-
traced trajectories for the particles using Runge-Kutta (RK) integration schemes. Either RK2, RK4, or
RK6 methods can be used. 

3. Nucleosynthesis.py

This interpolates T (temperature), Y_e  (electron fraction) and ρ (mass density) values along each
trajectory and outputs a H5 file of x, y, z, T, Y_e, ρ for the trajectories of all particles. These 
are also saved to a "results" folder, containing .txt files with this information for each particle.
These are ready to be inputted directly into nucleosynthesis networks.

Interpolations are performed by constructing RGIs for each iteration and refinement level for T, Y_e,
and ρ, which is again the slowest part. Once this is done, the code processes ~1000 particles per 
minute, as few interpolations are required.

.......................................................................................................................
                                                Testing (Speed, Complexity, Issues)
.......................................................................................................................                                               
It is of linear complexity, and is capable of tracing 50,000 particles in under 7 hours of computation (per node)
 - a drastic improvement from the ~500 tracers typically calculated within GRMHD simulations. This code is designed for
 execution on supercomputers, can also be executed on local systems by simply adjusting the number of parallel jobs.

Convergence tests display a ______ % agreement between trajectories computed within the simulation and with this code.

However, tracing can be sensitive to initial errors in velocity interpolation. Most particles are traced backwards to 
physically sensible regions, whilst some get "stuck" in low-velocity regions and form a "cloud" in final animations of 
the simulations. This is due to large timesteps, dt, between frames culminating in significant integration errors.  

To avoid this, only paths that begin within a certain radius R of the origin are saved into a separate file of 
filtered trajectories, which can be inputted into subsequent codes instead.
This method improves the agreement between pre-computed and post-processed tracers to ~ 90%.

.......................................................................................................................
                                                Installations / How to use    

1. Install Kuibit:
   Instructions on how to install Kuibit can be found here:  https://sbozzolo.github.io/kuibit/
   Kuibit is available in PyPI. To install it with pip, use the commands below:
       python 
       pip3 install kuibit 
   . 
       
   If you are operating on a cluster without an internet connection, install Kuibit into a folder
   on your local machine and copy it into the cluster. Then, simply run:
   
   module load hdf5
   module load python
   export PYTHONPATH=/path/to/kuibit/folder
   python
       
   . Now run the following commands to install Kuibit and all other required packages for the code 
   (though the others are generally already included as standard):
   
   import kuibit
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
   
Now you can run the codes by submiting batch files available in the repository. 

..........................................................................................................................
