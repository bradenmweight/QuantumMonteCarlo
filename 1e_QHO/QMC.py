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
    return -a + x1**2 * ( 2 * a**2 + 0.5)

# Define the Monte Carlo simulation function
def monte_carlo_simulation(a, num_samples, num_steps, step_size):
    total_energy = np.zeros( (num_samples,num_steps) )
    trajectory   = np.zeros( (num_samples,num_steps) )

    for sample in range(num_samples):
        x1 = random()*2-1 # Initial position for electron 1

        for step in range(num_steps):
            dx = (random()*2-1)*step_size
            x1_new = x1 + dx

            # Metropolis-Hastings acceptance criterion
            PSI_OLD = trial_wavefunction(x1, a)
            PSI_NEW = trial_wavefunction(x1_new, a)
            RATIO   = PSI_NEW / (PSI_OLD+1e-8)
            if RATIO > random():
                x1 = x1_new

            total_energy[sample,step] = local_energy(x1, a)
            trajectory[sample,step] = np.array([x1])

    NSTART = 500 # Skip the first {NSKIP} step to ensure equilibrium of the walkers
    return np.average( total_energy[:,NSTART:] ), np.var( total_energy[:,NSTART:] ), trajectory

# Main code
if __name__ == '__main__':
    num_samples = 2  # Number of samples
    num_steps = 10**4  # Number of Monte Carlo steps per sample
    step_size = 0.05  # Step size for the Monte Carlo steps


    """
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
    
    
    
    
    a_guess = 5.0  # Initial guess for parameter a
    dPARAM  = 0.001

    # Optimize the trial wavefunction using VQMC
    print( "\t a = %1.3f" % (a_guess) )
    NOPT = 50
    E   = np.zeros( NOPT )
    VAR = np.zeros( NOPT )
    a   = np.zeros( NOPT )
    
    for OPT in range(NOPT-1):
        a[OPT] = a_guess
        E[OPT], VAR[OPT], TRAJ = monte_carlo_simulation(a_guess, num_samples, num_steps, step_size)
        
        # Simplest optiimization -- BEST ONE. WHY ??? ~ BMW
        E_F, _, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        a_guess -= E_F - E[OPT]

        ### Do Newton-Raphson (secant) Method for parmeters
        ## NR Minimizing the Energy
        #E_F, _, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #GRAD = (E_F - E[OPT]) / dPARAM
        #a_guess -= E[OPT] / GRAD 

        ## NR Minimizing the Variance
        #_, VAR_F, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #GRAD = (VAR_F - VAR[OPT]) / dPARAM
        #a_guess -= VAR[OPT] / GRAD 

        print( OPT, "E (VAR) = %1.4f (%1.4f); a = %1.3f" % (E[OPT], VAR[OPT], a_guess) )

    plt.errorbar( np.arange(NOPT), a[:], yerr=VAR, ecolor="red", capsize=5, fmt="b-o" )
    plt.legend()
    plt.savefig("a_opt.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(NOPT), VAR[:], "-o", c='black' )
    plt.legend()
    plt.savefig("var_opt.jpg",dpi=300)
    plt.clf()

    plt.errorbar( np.arange(NOPT), E[:], yerr=VAR, ecolor="red", capsize=5, fmt="b-o" )
    plt.legend()
    plt.savefig("E_opt.jpg",dpi=300)
    plt.clf()