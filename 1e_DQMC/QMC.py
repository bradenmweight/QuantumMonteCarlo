import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def V(x, omega, model):
    if ( len(x.shape) == 2 ):
        x2 = np.einsum("wd,wd->w", x, x)
    else:
        x2 = x**2
    
    if ( model == "QHO" ):
        return 0.5 * omega**2 * x2 # Quantum Harmonic Oscillator



def DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model):

    # Initialize positions of the walkers
    positions = (np.random.uniform(size=(num_walkers,dimension))*2-1)*2

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):
        # Propagate the walkers
        positions += np.random.normal(0, 1, size=(len(positions), dimension))*np.sqrt(2*coupling*time_step)

        # Compute the potential energy for each walker
        potential_energies = V(positions, omega, model)

        # Compute the weights of the walkers
        alpha = 0.1 # Arbitrary parameter
        E_TRIAL = E_TRIAL + alpha * np.log( num_walkers / len(positions) )
        weights = np.exp(-time_step * (potential_energies - E_TRIAL))
        #print( "E Trial:", round(E_TRIAL,4), num_walkers, len(positions) )

        # Evaluate q = s + r
        s        = np.zeros(( len(weights) ), dtype=int)
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
num_walkers = 5000
num_steps = 500
time_step = 0.01
coupling = 0.5
omega = 1.0
E_TRIAL = 0.5
dimension = 6
model = "QHO" # "QHO"

# Run the DMC simulation for GS
positions = DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model)

# Print the results
print("\n\tNumber of Walkers:", len(positions))
print("\tAverage position:", np.average( positions, axis=0 ))
E_AVE = np.average( V(positions,omega, model), axis=0 )
E_STD = np.var( V(positions,omega, model), axis=0 )
print("\tAverage energy: %1.4f (%1.4f)" % (E_AVE,E_STD) )

# DMQ Result
PSI_0_DMQ, EDGES = np.histogram( positions[:,:].flatten(), bins=25 )
EDGES = (EDGES[:-1] + EDGES[1:])/2
PSI_0_DMQ = PSI_0_DMQ / np.linalg.norm( PSI_0_DMQ )

# Exact Result
if ( model == "QHO" ):
    E_EXACT = 0.500 * dimension
    X = np.linspace( -5,5,2000 )
    PSI_0_EXACT = np.exp( -X**2 / 2 ) + E_EXACT

# Compute Observables with DQMC Wavefunction
NX = len(EDGES)
dX = EDGES[1] - EDGES[0]
AVE_X  = np.sum( EDGES    * PSI_0_DMQ**2 ) * dX
AVE_X2 = np.sum( EDGES**2 * PSI_0_DMQ**2 ) * dX
print( "\t<x> = %1.4f, <x^2> = %1.4f, <x^2>-<x>^2 = %1.4f" % (AVE_X, AVE_X2, AVE_X2 - AVE_X**2 ) )
# Centered DFT for <p> and <p^2>
n, m   = np.arange(NX).reshape( (-1,1) ), np.arange(NX).reshape( (1,-1) )
W      = np.exp( -2j*np.pi * (m-NX//2) * (n-NX//2) / NX )
f_k    = (W @ PSI_0_DMQ).real * dX / np.sqrt( 2 * np.pi )
kmax   = 1 / 2 / dX # This is not angular frequency
k      = np.linspace( -kmax, kmax, NX )
dk     = k[1] - k[0]
AVE_P  = np.sum( k    * f_k**2 ) * dX
AVE_P2 = np.sum( k**2 * f_k**2 ) * dX
print( "\t<p> = %1.4f, <p^2> = %1.4f, <p^2>-<p>^2 = %1.4f" % (AVE_P, AVE_P2, AVE_P2 - AVE_P**2 ) )

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
plt.savefig("WAVEFUNCTION.jpg",dpi=300)

print("\n")