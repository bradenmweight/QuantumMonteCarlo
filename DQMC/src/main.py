import numpy as np
import sys
import os

from output import print_results, plot
from QMC import DMC

def get_polariton_parameters( PARAM, ARGS ):

    if ( PARAM["DO_POLARITON"] == True ):

        if ( len(ARGS) == 3 ):
            try:
                PARAM["CAVITY_COUPLING"] = float( ARGS[1] ) #A0 # a.u.
                PARAM["CAVITY_FREQ"]     = float( ARGS[2] ) #WC # eV
                PARAM["DATA_DIR"]       += "_A0_%1.4f_WC_%2.4f" % (PARAM["CAVITY_COUPLING"], PARAM["CAVITY_FREQ"] )
                PARAM["CAVITY_FREQ"]    /= 27.2114 # eV --> a.u.
            except:
                print("\n\tERROR!!! Something wrong with cavity parameters.\n")
                print( f"\t\tCoupling Strength: '{ARGS[1]}'\n\t\tCavity Frequency: '{ARGS[2]}'" )
                exit()
        else:
            print("\n\tWARNING!!! 'DO_POLARITON' was set to True but no parameters specified.")
            print("\tSetting cavity freqency and coupling to zero.\n")
            PARAM["CAVITY_COUPLING"] = 0.0 #A0 # a.u.
            PARAM["CAVITY_FREQ"]     = 0.0 #WC # a.u.
        
        # Normalize this thing...so user can do whatever they want
        PARAM["CAVITY_POLARIZATION"] = PARAM["CAVITY_POLARIZATION"] / np.linalg.norm(PARAM["CAVITY_POLARIZATION"])

        print( f"\t\tCoupling Strength: {round(PARAM['CAVITY_COUPLING'],3)} a.u.\n\t\tCavity Frequency: {round(PARAM['CAVITY_FREQ']*27.2114,3)} e.V.\n\t\tCavity Polarization: {PARAM['CAVITY_POLARIZATION']}" )
    
    return PARAM

def main():

    PARAM = get_Parameters( sys.argv[1:] )
    PARAM = get_polariton_parameters( PARAM, sys.argv[1:] ) # Check for polariton calculation and set parameters

    # Do equilibrium run
    positions, TRAJ, energy_traj, WFNs, PARAM = DMC(PARAM) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, WFNs, production_flag=False)

    # Do production run starting from equilibrated wavefunction
    positions, TRAJ, energy_traj, WFNs, PARAM = DMC(PARAM,positions=positions) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, WFNs, production_flag=True)

if ( __name__ == "__main__" ):
    sys.path.append( os.getcwd() )
    from Parameters import get_Parameters
    main()