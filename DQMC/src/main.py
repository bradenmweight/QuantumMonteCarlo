import numpy as np
import sys
import os

from output import print_results, plot
from potential import get_potential
from QMC import DMC

def check_for_polaritons( PARAM ):

    try:
        WC = PARAM["CAVITY_FREQ"]
        print("\n\tAdding polaritonic contributions.")
        print(f"\tWC   = {PARAM['CAVITY_FREQ']}")
        print(f"\tA0   = {PARAM['CAVITY_COUPLING']}")
        print(f"\tEPOL = {PARAM['CAVITY_POLARIZATION']}")
    except KeyError:
        PARAM["CAVITY_FREQ"] = None
    return PARAM

def get_PARAMS( ):

    PARAM = get_Parameters( sys.argv[1:] )
    return PARAM

def main():

    PARAM = get_PARAMS( )
    PARAM = check_for_polaritons( PARAM )
    

    if ( PARAM["CAVITY_FREQ"] is not None ):
        # Do equilibrium run
        positions, TRAJ, energy_traj, (EDGES, EL_WFN, PHOT_WFN), PARAM = DMC(PARAM) # This can be very easily parallelized.
        print_results( positions, energy_traj, PARAM )
        plot(positions, TRAJ, energy_traj, PARAM, EDGES, EL_WFN, PHOT_WFN, production_flag=False)

        # Do production run starting from equilibrated positions
        positions, TRAJ, energy_traj, (EDGES, EL_WFN, PHOT_WFN), PARAM = DMC(PARAM,positions=positions) # This can be very easily parallelized.
        print_results( positions, energy_traj, PARAM )
        plot(positions, TRAJ, energy_traj, PARAM, EDGES, EL_WFN, PHOT_WFN=PHOT_WFN, production_flag=True)

    else:
        # Do equilibrium run
        positions, TRAJ, energy_traj, (EDGES, WFN), PARAM = DMC(PARAM) # This can be very easily parallelized.
        print_results( positions, energy_traj, PARAM )
        plot(positions, TRAJ, energy_traj, PARAM, EDGES, WFN, production_flag=False)

        # Do production run starting from equilibrated positions
        positions, TRAJ, energy_traj, (EDGES, WFN), PARAM = DMC(PARAM,positions=positions) # This can be very easily parallelized.
        print_results( positions, energy_traj, PARAM )
        plot(positions, TRAJ, energy_traj, PARAM, EDGES, WFN, production_flag=True)

if ( __name__ == "__main__" ):
    sys.path.append( os.getcwd() )
    from Parameters import get_Parameters
    main()