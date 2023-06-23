import numpy as np

def get_Parameters( R_SEP=None ):
    PARAM = {}

    PARAM["num_walkers"] = 50000 # Averages and variance scale as 1/sqrt(N)

    # For equilibration
    PARAM["num_steps_equilibration"] = 500
    PARAM["time_step_equilibration"] = 0.1 # Choose such that: num_steps * time_step > 5
    PARAM["alpha_equilibration"] = 1.0 # Large enough to stem fluctuations in walkers
    # For production
    PARAM["num_steps_production"] = 5000
    PARAM["time_step_production"] = 0.005 # Choose such that: num_steps * time_step > 5
    PARAM["alpha_production"] = 0.01

    # Define System Parameters
    PARAM["R_NUC"] = np.zeros( (1,3) )
    PARAM["R_NUC"][0,0] = 0.0
    PARAM["Z_NUC"] = np.zeros( (1) )
    PARAM["Z_NUC"][0] = 2

    PARAM["dimension"]   = 3 # Can do many dimensions
    PARAM["particles"]   = 1 # Don't do more than ~3 particles. Need to optimize code first.
    PARAM["interacting"] = True

    PARAM["E_TRIAL_GUESS"] = -0.5 # Choose best guess

    return PARAM


def main():
    PARAMS = get_Parameters()

if ( __name__ == "__main__" ):
    main()