import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def V(x, omega, model, interacting):
    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V = np.zeros( (shapes[0], shapes[1]) )
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range(shapes[0]):
                    if ( p1 == p2 ): continue
                    R = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                    V[p1,:] += 1/np.abs(R)
        
        V[:,:] += 0.5 * omega * np.einsum( "pwd,pwd->pw", x, x )
        return np.average( V[:,:], axis=0 )
    else:
        r1_2 = x**2
        r2_2 = x**2
        V   = 0.5 * r1_2 + 0.5 * r2_2
        return V

def DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model, particles, interacting):

    # Initialize positions of the walkers
    positions = (np.random.uniform(size=(particles,num_walkers,dimension))*2-1)*2

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):
        # Propagate the walkers
        positions += np.random.normal(0, 1, size=(particles,len(positions[0,:]),dimension))*np.sqrt(2*coupling*time_step)

        # Compute the potential energy for each walker
        potential_energies = V(positions, omega, model, interacting)

        # Compute the weights of the walkers
        alpha = 0.1 # Arbitrary parameter
        E_TRIAL = E_TRIAL + alpha * np.log( num_walkers / len(positions[0,:]) )
        weights = np.exp(-time_step * (potential_energies - E_TRIAL))

        # Evaluate q = s + r
        s        = np.zeros(( len(positions[0,:]) ), dtype=int)
        RAND     = np.random.uniform( size=len(positions[0,:]) )
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

        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP).reshape( (particles,np.sum(s),dimension) )

        if ( step % 500 == 0 ):
            print ( f"Step = {step}, N = {len(positions[0,:,0])}" )
    return positions

# Parameters
num_walkers = 10000
num_steps = 2000
time_step = 0.01
coupling = 0.25
omega = 1.0
E_TRIAL = 0.5
model = "QHO"

dimension_list = [0,1,2,3,4,5,6]
particle_list  = [1,2,3,4,5]
interacting    = True

E_AVE = np.zeros( (10,10) )
E_VAR = np.zeros( (10,10) )
for dIND,d in enumerate(dimension_list):
    dimension = d+1
    for pIND,p in enumerate(particle_list):
        particles = p+1
        # Run the DMC simulation for GS
        positions = DMC(num_walkers, num_steps, time_step, coupling, omega, E_TRIAL, dimension, model, particles, interacting)

        # Print the results
        print("\n\tNumber of Walkers:", len(positions[0,:]))
        print("\tAverage position:", np.average( positions, axis=(0,1) ) )
        E_AVE[d,p] = np.average( V(positions, omega, model, interacting) )
        E_VAR[d,p] = np.var( V(positions,omega, model, interacting) )
        print(f"\tAverage energy: {E_AVE[d,p]} " )
        print(f"\tVAR     energy: {E_VAR[d,p]} " )

        # DMQ Result
        PSI_0_DMQ_P0, EDGES = np.histogram( positions[:,:].flatten(), bins=25 )
        EDGES = (EDGES[:-1] + EDGES[1:])/2
        PSI_0_DMQ_P0 = PSI_0_DMQ_P0 / np.linalg.norm( PSI_0_DMQ_P0)

        # Exact Result
        if ( model == "QHO" ):
            E_EXACT = 0.500 * dimension * particles # Many Non-interacting Particles 
            X = np.linspace( -5,5,2000 )
            PSI_0_EXACT = np.exp( -X**2 / 2 ) + E_EXACT

        # Compute Observables with DQMC Wavefunction
        NX = len(EDGES)
        dX = EDGES[1] - EDGES[0]
        AVE_X  = np.sum( EDGES    * PSI_0_DMQ_P0**2 ) * dX
        AVE_X2 = np.sum( EDGES**2 * PSI_0_DMQ_P0**2 ) * dX
        print( "\t<x> = %1.4f, <x^2> = %1.4f, <x^2>-<x>^2 = %1.4f" % (AVE_X, AVE_X2, AVE_X2 - AVE_X**2 ) )

        # Plot the Results
        plt.plot( EDGES, PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0) + E_AVE[d,p], "-o", c="red", label="DQMQ" )
        plt.plot( X, PSI_0_EXACT, "-", c="black", label="EXACT Non-interacting" )
        plt.plot( X, V(X,omega,model,interacting), label="V(x)" )
        plt.legend()
        MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
        plt.xlim( -MAX_X*1.,MAX_X*1.)
        plt.ylim( 0, 1.5*(np.max(PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0)) + E_AVE[d,p]) )
        plt.xlabel("Position, X",fontsize=15)
        plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
        plt.title(f"Diffusion QMC Results: {model} {dimension}D\n Interacting: {interacting}",fontsize=15)
        plt.savefig(f"WAVEFUNCTION_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
        plt.clf()

        print("\n")

    np.savetxt("E_AVE.dat", E_AVE)
    np.savetxt("E_VAR.dat", E_VAR)

    for p in range( 5 ):
        plt.plot( dimension_list, E_AVE[:,p], label=f"{p} Particles" )
        plt.xlabel("Dimension, d",fontsize=15)
        plt.ylabel("Ground State Energy",fontsize=15)
        plt.savefig(f"GS_ENERGY_d_p.jpg",dpi=300)
        plt.clf()