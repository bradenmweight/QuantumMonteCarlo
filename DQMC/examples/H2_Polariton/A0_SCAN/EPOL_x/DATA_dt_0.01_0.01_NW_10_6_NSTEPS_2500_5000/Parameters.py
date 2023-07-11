import numpy as np
import sys

def get_Parameters( ARGS ):
    PARAM = {}

    PARAM["DATA_DIR"] = "PLOTS_DATA"

    PARAM["num_walkers"] = 10**6

    # For equilibration
    PARAM["num_steps_equilibration"] = 5000
    PARAM["time_step_equilibration"] = 0.01
    PARAM["alpha_equilibration"] = PARAM["time_step_equilibration"]
    # For production
    PARAM["num_steps_production"] = 5000
    PARAM["time_step_production"] = 0.01
    PARAM["alpha_production"] = PARAM["time_step_production"]

    # Define System Parameters
    PARAM["R_NUC"] = np.zeros( (2,3) )
    PARAM["R_NUC"][0,0] = 0.0
    if ( len(ARGS) >= 1 ):
        PARAM["R_NUC"][1,0] = float( ARGS[0] )
        PARAM["DATA_DIR"] = "DATA_R_%1.3f" % PARAM["R_NUC"][1,0]
    else:
        PARAM["R_NUC"][1,0] = 1.0
    PARAM["R_NUC"] -= np.average( PARAM["R_NUC"], axis=0 ) # Shift to COM
    PARAM["Z_NUC"] = np.zeros( (2) )
    PARAM["Z_NUC"][0] = 1
    PARAM["Z_NUC"][1] = 1

    PARAM["dimension"]   = 3 # Can do many dimensions
    PARAM["particles"]   = 2 # Don't do more than ~3 particles. Need to optimize code first.
    PARAM["interacting"] = True

    PARAM["E_TRIAL_GUESS"] = -1.0 # Choose best guess


    ### BELOW ARE FOR POLARITON CALCULATIONS ###

    PARAM["DO_POLARITON"] = True
    PARAM["CAVITY_POLARIZATION"] = np.array([1,0,0]) # Will normalize later

    return PARAM


def main():
    PARAMS = get_Parameters()

if ( __name__ == "__main__" ):
    main()