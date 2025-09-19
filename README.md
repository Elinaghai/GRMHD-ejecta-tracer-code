
# GRMHD-Simulation-Post-Processing-tracer-code

When neutron stars collide and coalesce in a Binary Neutron Star (BNS) merger, a non-negligible fraction of
neutron-rich, hot and dense matter is ejected into space. Due to the extremely high abundance of neutrons
in these ejecta, and the rapid cooling/expansion that they undergo as they are expelled from the remnant, 
BNS ejecta provide an environment ripe for R-process nucleosynthesis.

Typical computational analysis of this phenomenon consists of computing so-called "tracers" within GRMHD simulations 
of BNS mergers. These are Lagrangian particles which represent the motion of fluid elements within the ejecta, whose 
trajectories (and associated quantities relevant for nucleosynthesis) are used to compute the final abundances 
resulting from nucleosynthesis in the ejecta (utlising nucleosynthesis networks).

However, computing tracers within simulations is restrictive. This method only allows for a limited number of tracers
to be calculated, and does not a) allow for retrospective analysis of previous simulations, nor
 b) allow for multiple ejecta types to be considered. 
Fundamentally, most GRMHD simulations cannot provide sufficient tracers to gain an appreciable picture of the ejecta.

This produces the need for a system of codes which generate, select and trace lagrangian particles outside 
the simulation itself- which is the result of this work. It can select all types of ejecta, and traces particles at 
a rate of ~150 particles per minute, per node, for a dataset of ~250 iterations. 


## Acknowledgements


 This code extensively uses [Kuibit](https://sbozzolo.github.io/kuibit/index.html#), a library of post-processing tools to handle data from numerical relativity simulations. I thus acknowledge Gabriele Bozzola, the author of this excellent resource.

I would also like to acknowledge the work of Guiseppe Rivieccio at the University of Valencia, whose contributions and ideas made this work possible.
Last but not least, I would like to acknowledge the contributions of my supervisor Milton Ruiz, whose insights shaped this project.
## Author

- [@Elinaghai](https://github.com/Elinaghai)
You may contact me at eg1022@ic.ac.uk for any enquiries regarding these codes.


## Features

grid_code.py :

- Outputs a dictionary of iterations and corresponding time values for each output of simulation data, later used in tracer_code.py

- Resamples grid data in cases where multiple patches are present, accounting for setups with nested grids following each star

- Resampling grid data enables us to avoid defaulting to coarser grids where Kuibit's get_level method fails (due to the presence of multiple patches). This reduces errors in any interpolated values calculated.


tracer_code.py :

- Generates particles at last simulation frame and selects those which are unbound (-1-u_0>0) and moving radially outwards (v_r>0), and thus represent fluid elements in ejecta

- Regular Grid Interpolators (RGIs) constructed for each refinement level at each iteration and are stored in a dictionary

- Traces particles backwards throughout the simulation to compute trajectories via interpolating with RGIs to find velocities and using Runge-Kutta (RK) integration methods

- RK4 or RK6 backwards integration methods available

- 100 (RK6) to 150 (RK4) particles computed per minute during tracing


nucleosynthesis_code.py

- Produces T (temperature), Y_e  (electron fraction) and ρ (mass density) values along each particle's trajectory and outputs a H5 file of x, y, z, T, Y_e, ρ for the paths of all particles. Values are interpolated from grid data

- Produces .txt files of x, y, z, T, Y_e and ρ, ready to be directly inputted into nucleosynthesis networks

animation_code.py

- Produces a .gif file illustrating the particles' motion (whose trajectories begin within a distance R of the origin)

- Saves individual animation frames into a separate directory







## Installation
These codes are designed for execution on a supercomputer cluster.

Firstly, install Kuibit onto the cluster by running: 

```bash
  python
  pip3 install kuibit
```
If you are operating on a cluster without an internet connection, install Kuibit into a folder on your local machine and copy it into the cluster. After this, simply run:

```bash
  module load hdf5
  module load python
  export PYTHONPATH=/path/to/kuibit/folder
```
In either case, you will need to load in all packages before executing the code by running the following:

```bash
python
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

```

. 
## Deployment

These codes may be executed by submitting the batch files available in the repository. In the case of the MareNostrum supercomputer, the tracer code (for example) can be run using
```bash
sbatch job_script_tracer.pbs
```
grid_code.py must be executed first, followed by tracer_code.py. After this, nucleosynthesis_code.py and animation_code.py can be executed in any order.
