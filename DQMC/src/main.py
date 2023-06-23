import numpy as np
import sys
import os

from output import print_results, plot
from potential import get_potential
from QMC import DMC


def main( DATA_DIR=None ):
    if ( DATA_DIR is not None ):
        PARAM = get_Parameters( R_SEP=float(DATA_DIR) )
        PARAM["DATA_DIR"] = DATA_DIR
    else:
        PARAM = get_Parameters( )
        PARAM["DATA_DIR"] = None

    # Do equilibrium run
    positions, TRAJ, energy_traj, (EDGES, WFN) = DMC(PARAM) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, EDGES, WFN, production_flag=False)

    # Do production run starting from equilibrated positions
    positions, TRAJ, energy_traj, (EDGES, WFN) = DMC(PARAM,positions=positions) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, EDGES, WFN, production_flag=True)

if ( __name__ == "__main__" ):
    sys.path.append( os.getcwd() )
    from Parameters import get_Parameters
    if ( len(sys.argv) > 1 ):
        DATA_DIR = sys.argv[1]
        main(DATA_DIR=DATA_DIR)
    else:
        main()