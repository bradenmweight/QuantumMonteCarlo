import numpy as np

from output import print_results, plot
from potential import get_potential
from QMC import DMC

def get_Parameters():

    PARAM = {}

    # Parameters
    PARAM["num_walkers"] = 10**3
    PARAM["num_steps"] = 10
    PARAM["time_step"] = 0.05 # Choose such that: num_steps * time_step > 5

    PARAM["R_NUC"] = np.zeros( (2,3) ) # One H, One Li, 3 dimensions
    PARAM["R_NUC"][0,0] = 0.0 # Li
    PARAM["R_NUC"][0,0] = 1.65 # H
    PARAM["Z_NUC"] = np.zeros( (2) )  # One H, One Li
    PARAM["Z_NUC"][0] = 3  # One Li
    PARAM["Z_NUC"][1] = 1  # One H

    PARAM["dimension"]   = 3 # Can do many dimensions
    PARAM["particles"]   = 4 # Don't do more than ~3 particles. Need to optimize code first.
    PARAM["interacting"] = True

    PARAM["E_TRIAL_0"] = -8.0 # Choose best guess

    return PARAM

def main():
    PARAM = get_Parameters()
    positions, TRAJ, energy_traj = DMC(PARAM) # This can be very easily parallelized.
    print_results( positions, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM)

if ( __name__ == "__main__" ):
    main()