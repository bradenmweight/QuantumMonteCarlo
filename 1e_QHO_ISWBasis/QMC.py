import numpy as np
from random import random
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

# Define the potential energy function for the infinite square well
def potential_energy(x1):
    return 0.5 * x1**2

# Define the trial wavefunction
def trial_wavefunction(x1, PARAMS):
    # <x|\psi> --> For a single x
    a,b,n = PARAMS
    #return a + b * np.cos( 2 * np.pi * x1 * n )
    return np.exp( -a * x1**2 )

# Define the local energy function with electron-electron interactions
def local_energy(x1, PARAMS):
    # H <x|\psi> = E <x|\psi> --> For a single x (i.e., no integral)
    
    # Do this numerically (faster would be to use analytic gradients)
    X = np.linspace( x1-1,x1+1,10 )
    f_X = trial_wavefunction( X, PARAMS )
    f_X_smooth = UnivariateSpline(X,f_X,s=0,k=3)
    T = f_X_smooth.derivative(n=2)(x1)
    #GRAD_f_x = np.gradient(f_X, X)
    #T = -0.5 * np.gradient( GRAD_f_x, X )
    #T = T[ len(X)//2 ]
    return T + potential_energy(x1)

# Define the Monte Carlo simulation function
def monte_carlo_simulation(PARAMS, num_samples, num_steps, step_size):
    total_energy = np.zeros( (num_samples,num_steps) )
    trajectory   = np.zeros( (num_samples,num_steps) )

    for sample in range(num_samples):
        x1 = random()*2-1 # Initial position for electron 1

        for step in range(num_steps):
            dx = (random()*2-1)*step_size
            x1_new = x1 + dx

            # Metropolis-Hastings acceptance criterion
            PSI_OLD = trial_wavefunction(x1, PARAMS)
            PSI_NEW = trial_wavefunction(x1_new, PARAMS)
            RATIO   = PSI_NEW / (PSI_OLD+1e-8)
            if RATIO > random():
                x1 = x1_new

            total_energy[sample,step] = local_energy(x1, PARAMS)
            trajectory[sample,step] = np.array([x1])

    NSTART = 500 # Skip the first {NSKIP} step to ensure equilibrium of the walkers
    return np.average( total_energy[:,NSTART:] ), np.var( total_energy[:,NSTART:] ), trajectory

# Main code
if __name__ == '__main__':
    num_samples = 2  # Number of samples
    num_steps = 10**3  # Number of Monte Carlo steps per sample
    step_size = 0.05  # Step size for the Monte Carlo steps
    NOPT = 10

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
    
    
    
    
    INIT_PARAMS = np.array([0.5, 0.5, 1.0])  # Initial guess for all parmeters
    dPARAM      = 0.001 # For derivatives

    # Optimize the trial wavefunction using VQMC
    print( "\t a = %1.3f" % (INIT_PARAMS[0]) )
    print( "\t b = %1.3f" % (INIT_PARAMS[1]) )
    print( "\t n = %1.3f" % (INIT_PARAMS[2]) )
    E         = np.zeros( NOPT )
    VAR       = np.zeros( NOPT )
    PARAMS    = np.zeros( (NOPT, len(INIT_PARAMS)) )
    DAMP      = 1.0

    PARAMS[0] = INIT_PARAMS
    
    for OPT in range(NOPT-1):
        E[OPT], VAR[OPT], TRAJ = monte_carlo_simulation(PARAMS[OPT,:], num_samples, num_steps, step_size)
        
        # Simplest optiimization -- BEST ONE. WHY ??? ~ BMW
        E_F = np.zeros(3)
        E_F[0], _, _ = monte_carlo_simulation(PARAMS[OPT,:] + dPARAM*np.array([1,0,0]), num_samples, num_steps, step_size)
        #E_F[1], _, _ = monte_carlo_simulation(PARAMS[OPT,:] + dPARAM*np.array([0,1,0]), num_samples, num_steps, step_size)
        #E_F[2], _, _ = monte_carlo_simulation(PARAMS[OPT,:] + dPARAM*np.array([0,0,1]), num_samples, num_steps, step_size)
        PARAMS[OPT,:] -= DAMP * (E_F[:] - E[OPT])

        ### Do Newton-Raphson (secant) Method for parmeters
        ## NR Minimizing the Energy
        #E_F, _, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #GRAD = (E_F - E[OPT]) / dPARAM
        #a_guess -= E[OPT] / GRAD 

        ## NR Minimizing the Variance
        #_, VAR_F, _ = monte_carlo_simulation(a_guess + dPARAM, num_samples, num_steps, step_size)
        #GRAD = (VAR_F - VAR[OPT]) / dPARAM
        #a_guess -= VAR[OPT] / GRAD 

        print( OPT, "E (VAR) = %1.4f (%1.4f);\n\ta = %1.3f b = %1.3f n = %1.3f" % (E[OPT], VAR[OPT], PARAMS[OPT,0], PARAMS[OPT,1], PARAMS[OPT,2]) )

    plt.errorbar( np.arange(NOPT), PARAMS[:,0], yerr=VAR, ecolor="black", capsize=5, fmt="b-o", label="a" )
    plt.errorbar( np.arange(NOPT), PARAMS[:,1], yerr=VAR, ecolor="black", capsize=5, fmt="r-o", label="b" )
    plt.errorbar( np.arange(NOPT), PARAMS[:,2], yerr=VAR, ecolor="black", capsize=5, fmt="g-o", label="n" )
    plt.legend()
    plt.savefig("params_opt.jpg",dpi=300)
    plt.clf()

    plt.plot( np.arange(NOPT), VAR[:], "-o", c='black' )
    plt.legend()
    plt.savefig("var_opt.jpg",dpi=300)
    plt.clf()

    plt.errorbar( np.arange(NOPT), E[:], yerr=VAR, ecolor="red", capsize=5, fmt="b-o" )
    plt.legend()
    plt.savefig("E_opt.jpg",dpi=300)
    plt.clf()

    X = np.linspace(-2,2,2000)
    V = potential_energy(X)
    PSI = trial_wavefunction(X,PARAMS[-1,:])
    plt.plot( X, V, c="black", lw=4, label="V(x)" )
    plt.plot( X, PSI + E[-1], label="WFN" )
    plt.savefig("WFN.jpg",dpi=300)
    plt.clf()