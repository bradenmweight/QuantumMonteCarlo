import numpy as np
from random import random
from matplotlib import pyplot as plt

# Define the potential energy function for the infinite square well
def potential_energy(x):
    return 0.5 * x**2

# Define the trial wavefunction
def trial_wavefunction(x1, a):
    return np.exp(-a * x1**2)

# Define the local energy function with electron-electron interactions
def local_energy(x1, a):
    # H <x|\psi| = E<x|\psi> --> For a single x
    return 0.5 * x1**2 + a * (1 - 2 * a * x1**2)

# Define the Monte Carlo simulation function
def monte_carlo_simulation(a, num_samples, num_steps, step_size):
    total_energy = np.zeros( (num_samples,num_steps) )
    trajectory   = np.zeros( (num_samples,num_steps) )

    #    for sample in range(num_samples):
    x1 = np.random.uniform(size=(num_samples))*2-1 # Initial position for electron 1
    x1 *= 5 # Sample a large configuration space

    for step in range(num_steps):
        dx = (np.random.uniform(size=(num_samples))*2-1)*step_size
        x1_new = x1 + dx

        # Metropolis-Hastings acceptance criterion
        PSI_OLD = trial_wavefunction(x1, a)
        PSI_NEW = trial_wavefunction(x1_new, a)
        RATIO   = PSI_NEW / (PSI_OLD)
        RAND = np.random.uniform( size=(num_samples) )
        x1 = np.array([ x1_new[w] if RATIO[w] > RAND[w] else x1[w] for w in range(len(x1)) ])

        total_energy[:,step] = local_energy(x1, a)
        trajectory[:,step]   = x1

    NSTART = 10 # Skip the first {NSKIP} step to ensure equilibrium of the walkers
    return np.average( total_energy[:,NSTART:] ), np.var( total_energy[:,NSTART:] ), trajectory

# Main code
if __name__ == '__main__':
    num_samples = 100000  # Number of samples
    num_steps = 20 # Number of Monte Carlo steps per sample
    step_size = 0.005  # Step size for the Monte Carlo steps


    """
    # DO A SCAN TO CHECK THE VARIANCE PLOT -- VERY CLEAR
    a_LIST    = np.arange( 0.25, 0.80, 0.05 )
    E   = np.zeros( len(a_LIST) )
    VAR = np.zeros( len(a_LIST) )
    for aIND,a_guess in enumerate(a_LIST):
        E[aIND], VAR[aIND], TRAJ = monte_carlo_simulation(a_guess, num_samples, num_steps, step_size)
        print( "\tE = %1.4f a = %1.4f" % (E[aIND], a_guess) )

    plt.errorbar( a_LIST, E, yerr=VAR, ecolor="red", capsize=5, fmt="b-o" )
    plt.xlabel("Parameter $a$", fontsize=18)
    plt.ylabel("Energy (a.u.)", fontsize=18)
    plt.savefig("a.jpg",dpi=300)
    """
    
    
    
    
    a_guess = 1.0  # Initial guess for parameter a
    THRESH  = 1e-3
    dPARAM  = 0.001

    # Optimize the trial wavefunction using VQMC
    print( "\t a = %1.3f (Guess)" % (a_guess) )
    NOPT = 200
    E   = []
    VAR = []
    a   = []
    E0, VAR0, TRAJ = monte_carlo_simulation(a_guess, num_samples, num_steps, step_size)
    E.append(E0)
    VAR.append(VAR0)
    a.append(a_guess)
    E_F, _, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
    print( 0, "E (VAR) = %1.4f (%1.4f); a = %1.3f" % (E[0], VAR[0], a_guess) )
    a_guess -= E_F - E[0]

    for OPT in range(1,NOPT):
        Ej, VARj, TRAJ = monte_carlo_simulation(a_guess, num_samples, num_steps, step_size)
        E.append(Ej)
        VAR.append(VARj)

        # Simplest possible optiimization
        E_F, VAR_F, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #a_guess -= E_F - E[-1] # Very unstable, but sometimes gets right answer
        DAMP     = 1 - np.abs(VAR_F - VAR[-1]) / VAR[-1]
        print ( "\tDAMP:", DAMP )
        a_guess -= DAMP * (VAR_F - VAR[-1]) # Somehow more stable...

        ## NR almost never converges...Not sure why
        # Newton-Raphson (First-order Derivative --> Secant Method)
        #E_F, VAR_F, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #dEda      = (E_F - E[OPT]) / dPARAM
        #a_guess -= E[OPT] / dEda
        #dVda      = (VAR_F - VAR[OPT]) / dPARAM
        #a_guess -= VAR[OPT] / dVda


        a.append(a_guess)
        print( OPT, "E (VAR) = %1.4f (%1.6f); a = %1.6f" % (E[-1], VAR[-1], a_guess) )
        if ( OPT > 5 and abs(a[-1] - a[-2]) < THRESH ):
            print("\tFinished. Exitting optimization.")
            break
        elif ( OPT > NOPT ):
            print(f"\tWATNING!!!!\n\tOptimization not converged.\n\tQutting loop after {NOPT} step.")


    plt.plot( np.arange(len(a)), a[:], "-o", c='black' )
    plt.xlabel("Optimization Step",fontsize=15)
    plt.ylabel("Parameter a",fontsize=15)
    plt.savefig("a_opt.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(len(a)), VAR[:], "-o", c='black' )
    plt.xlabel("Optimization Step",fontsize=15)
    plt.ylabel("Parameter a",fontsize=15)
    plt.savefig("var_opt.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(len(a)), E[:], "-o", c='black' )
    plt.xlabel("Optimization Step",fontsize=15)
    plt.ylabel("Parameter a",fontsize=15)
    plt.savefig("E_opt.jpg",dpi=300)
    plt.clf()