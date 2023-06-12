import numpy as np
from matplotlib import pyplot as plt
import random
from numba import jit

def V(x, omega, model):
    if ( len(x.shape) == 2 ):
        x2 = np.einsum("wd,wd->w", x, x)
    else:
        x2 = x**2
    
    if ( model == "QHO" ):
        return 0.5 * omega**2 * x2 # Quantum Harmonic Oscillator
    elif ( model == "ISW" ):
        return np.array([ 0 if (-1 < np.sqrt(x2[w]) < 1) else \
                          1000 for w in range(len(x2))  ]) # Infinite Square Well
    elif ( model == "ISW_STEP" ):
        V = np.zeros( (len(x2)) )
        for w in range(len(x2)):
            if ( -1 < np.sqrt(x2[w]) < 0 ):
                V[w] = 0
            elif ( 0 < np.sqrt(x2[w]) < 1 ):
                V[w] = 1
            else:
                V[w] = 1000
        return V



def DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model):
    # Initialize positions of the walkers
    #positions = np.random.randn(num_walkers, dimension)
    #positions = np.array([ [(random.random()*2-1)*3 for d in range(dimension)] for w in range(num_walkers) ] )
    positions = (np.random.uniform(size=(num_walkers,dimension))*2-1)*1

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):
        # Propagate the walkers
        positions += np.random.normal(0, 1, size=(len(positions), dimension))*np.sqrt(2*coupling*time_step)
        #positions += np.array([ [random.gauss(0, 1)*np.sqrt(2*coupling*time_step) for d in range(dimension)] for w in range(len(positions)) ] )

        # Compute the potential energy for each walker
        potential_energies = V(positions, omega, model)

        # Compute the weights of the walkers
        alpha = 0.5 # Arbitrary parameter
        E_TRIAL = E_TRIAL + alpha * np.log( num_walkers / len(positions) )
        weights = np.exp(-time_step * (potential_energies - E_TRIAL))
        #print( "E Trial:", E_TRIAL, num_walkers, len(positions) )

        # Evaluate q = s + r
        s        = np.zeros(( len(weights) ), dtype=int)
        #RAND     = np.array([ random.random() for w in range(len(positions)) ])
        RAND     = np.random.uniform(size=len(positions))
        IND_ZERO = weights < RAND  # s --> 0
        IND_2    = weights - 1 > RAND # s --> 2
        s[ IND_ZERO ]  = -1 # Set to dummy labels
        s[ IND_2 ]     = -2 # Set to dummy labels
        s[ s == 0 ]    = 1 # Keep these
        s[ IND_ZERO ]  = 0 # Kill these
        s[ IND_2 ]     = 2 # Duplicate these

        if ( np.sum( s ) < 1 ):
            print( "WARNING !!!!" )
            print( "Number of walkers went to zero..." )
            exit()

        positions = np.repeat( positions, s, axis=0 )

        if ( step % 500 == 0 ):
            print ( f"Step = {step}, N = {len(positions)}" )
    return positions

# Parameters
num_walkers = 10000
num_steps = 2000
time_step = 0.01
coupling = 0.5
omega = 1.0
E_TRIAL = 1.5
dimension = 10
model = "QHO" # "QHO", "ISW", "ISW_STEP"

# Run the DMC simulation
positions = DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model)

# Print the results
print("Number of Walkers:", len(positions))
print("Average position:", np.average( positions, axis=0 ))
E_AVE = np.average( V(positions,omega, model), axis=0 )
E_STD = np.std( V(positions,omega, model), axis=0 )
print("Average energy: %1.4f (%1.4f)" % (E_AVE,E_STD) )

# DMQ Result
PSI_0_DMQ, EDGES = np.histogram( positions[:,:].flatten(), bins=25 )
EDGES = (EDGES[:-1] + EDGES[1:])/2
PSI_0_DMQ = PSI_0_DMQ / np.linalg.norm( PSI_0_DMQ )

# Exact Result
if ( model == "QHO" ):
    E_EXACT = 0.500 * dimension
    X = np.linspace( -5,5,2000 )
    PSI_0_EXACT = np.exp( -X**2 / 2 ) + E_EXACT
elif ( model == "ISW" ):
    L = 2
    E_EXACT = np.pi**2 / (2 * L**2)
    X = np.linspace( -1,1,2000 )
    PSI_0_EXACT = np.cos( np.pi * X / L ) + E_EXACT
elif ( model == "ISW_STEP" ):
    X = np.linspace( -1,1,2000 )
    PSI_0_EXACT = X*float("Nan")

# Plot the Results
plt.plot( EDGES, PSI_0_DMQ / np.max(PSI_0_DMQ) + E_AVE, "-o", c="red", label="DQMQ" )
plt.plot( X, PSI_0_EXACT, "-", c="black", label="EXACT" )
plt.plot( X, V(X,omega,model), label="V(x)" )
plt.legend()
MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
plt.xlim( -MAX_X*1.5,MAX_X*1.5)
plt.ylim( 0, 1.5*(np.max(PSI_0_DMQ / np.max(PSI_0_DMQ)) + E_AVE) )
plt.xlabel("Position, X",fontsize=15)
plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
plt.title(f"Diffusion QMC Results: {model} {dimension}D",fontsize=15)
plt.savefig("weights.jpg",dpi=300)
