import numpy as np
import sys

def get_Parameters( ARGS ):
    PARAM = {}

    PARAM["DATA_DIR"] = "PLOTS_DATA"

    PARAM["num_walkers"] = 10**5 # Accuracy scales as 1/sqrt(N)

    # For equilibration
    PARAM["num_steps_equilibration"] = 500
    PARAM["time_step_equilibration"] = 0.1 # Choose such that: num_steps * time_step > 5
    PARAM["alpha_equilibration"] = 0.1 # Large enough to stem fluctuations in walkers
    # For production
    PARAM["num_steps_production"] = 500
    PARAM["time_step_production"] = 0.1 # Choose such that: num_steps * time_step > 5
    PARAM["alpha_production"] = 0.1

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
    if ( len(ARGS) >= 1 ):
        PARAM["CAVITY_FREQ"]     = 3.0/27.2114 # a.u.
        PARAM["CAVITY_COUPLING"] = float( ARGS[1] ) #A0 # a.u. # A0
        PARAM["DATA_DIR"]       += "_A0_%1.3f" % PARAM["CAVITY_COUPLING"]
    else:
        PARAM["CAVITY_COUPLING"] = 0.0 #A0 # a.u. # A0
        PARAM["CAVITY_FREQ"]     = 3.0/27.2114 # a.u.

    PARAM["CAVITY_POLARIZATION"] = np.array([0,1,0])
    
    # Normalize this thing...so user can do whatever they want
    PARAM["CAVITY_POLARIZATION"] = PARAM["CAVITY_POLARIZATION"] / np.linalg.norm(PARAM["CAVITY_POLARIZATION"])


    return PARAM


def main():
    PARAMS = get_Parameters()

if ( __name__ == "__main__" ):
    main()