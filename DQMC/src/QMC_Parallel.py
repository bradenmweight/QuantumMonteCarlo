import numpy as np
import multiprocessing as mp

from potential_parallel import get_potential

def DMC_step( WALKER_IND, PARAM ): # This is parallelized

    POSITION = PARAM["POSITIONS"][:,WALKER_IND,:]

    # Propagate the walker
    if ( PARAM["EQUILIBRATION_FLAG"] ):
        positions += np.random.normal(0, np.sqrt(PARAM["time_step_equilibration"]), size=() )
        if ( PARAM["DO_POLARITON"] ):
            PARAM["QC"] += np.random.normal( 0, np.sqrt(PARAM["time_step_equilibration"]) )
    else:
        positions += np.random.normal(0, np.sqrt(PARAM["time_step_production"]) )
        if ( PARAM["DO_POLARITON"] ):
            PARAM["QC"] += np.random.normal( 0, np.sqrt(PARAM["time_step_production"]) )



        #### ENDED HERE ####



        # Compute the potential energy for each walker
        potential_energy = get_potential(POSITION,PARAM)

        # Compute the weights of the walkers
        PARAM["ENERGY_TRIAL"] += alpha * np.log( num_walkers / len(positions[0,:]) )
        dE = potential_energy - PARAM["ENERGY_TRIAL"]

        # POW_THRESH = 7 # This is somewhat arbitrary
        # if ( (-time_step * dE).any() > POW_THRESH ):
        #     INDICES = [ j for j,de in enumerate(dE) if (-time_step * de) > POW_THRESH ]
        #     dE[ INDICES ] = -7 / time_step

        weights  = np.exp( -time_step * dE )


        # Evaluate q = s + r
        s        = np.ones(( len(positions[0,:]) ), dtype=int)
        RAND     = np.random.uniform( size=len(positions[0,:]) )
        IND_ZERO = weights < RAND  # s --> 0
        IND_2    = weights - 1 > RAND # s --> 2
        s[ IND_ZERO ]  = -1 # Set to dummy labels for later killing
        s[ IND_2 ]     = -2 # Set to dummy labels for later doubling
        s[ s == 0 ]    = 1 # Keep everything else as is
        s[ IND_ZERO ]  = 0 # Kill these
        s[ IND_2 ]     = 2 # Duplicate these --> Later, try np.floor(weights - 1) if (weights - 1 >= 1)
                                # This would allow to triple, quadrupole, etc. the walkers

        if ( np.sum( s ) < 1 ):
            print( "WARNING !!!!" )
            print( "Number of walkers went to zero..." )
            exit()
        #elif ( np.sum( s ) > 10**7 / len(positions[:]) / len(positions[0,:] / len(positions[0,0,:])) ):
        #elif ( np.sum( s ) > 10**7 / len(positions[0,0,:]) ):
        #    print( "WARNING !!!!" )
        #    print( "Too many walkers were spawned...", np.sum( s ) )
        #    exit()

        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP).reshape( (particles,np.sum(s),dimension) )
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] = np.repeat( PARAM["QC"], s[:] )

        if ( len(positions[0]) > 100 ):
            trajectory[:,:,:,step] = positions[:,:100,:]

        if ( step == 0 ):
            NEDGES = 1000
            XMIN   = np.min( R_NUC[:,:]*5 )
            XMAX   = np.max( R_NUC[:,:]*5 )
            XMAX   = np.max( [abs(XMIN),XMAX] )
            XMIN   = -XMAX
            EL_WFN = np.zeros( ( NEDGES, dimension ) )
            for d in range( dimension ):
                EL_WFN[:,d], EDGES = np.histogram( positions[:,:,d].flatten(), bins=np.linspace(XMIN,XMAX,NEDGES+1) )
            if ( PARAM["DO_POLARITON"] == True ):
                PHOT_WFN, _ = np.histogram( PARAM["QC"], bins=np.linspace(XMIN,XMAX,NEDGES+1) )
        else:
            for d in range( dimension ):
                TMP     = np.histogram( positions[:,:,d].flatten(), bins=np.linspace(XMIN,XMAX,NEDGES+1) )
                EL_WFN[:,d] += TMP[0]
            if ( PARAM["DO_POLARITON"] == True ):
                TMP = np.histogram( PARAM["QC"], bins=np.linspace(XMIN,XMAX,NEDGES+1) )
                PHOT_WFN += TMP[0]
        
    if ( PARAM["DO_POLARITON"] == True ):
        return positions, trajectory, energy_traj, (EDGES, EL_WFN, PHOT_WFN), PARAM
    else:
        return positions, trajectory, energy_traj, (EDGES, EL_WFN), PARAM

def prepare( PARAM ):

    particles = PARAM["particles"]
    num_walkers = PARAM["num_walkers"]
    dimension = PARAM["dimension"]
    R_NUC = PARAM["R_NUC"]

    try:
        TMP = PARAM["POSITIONS"]
        # Sample from equilibration wavefunction
        for p in range( particles ):
            for dim in range( dimension ):
            PARAM["POSITIONS"][p,:,dim] = random.choices(mylist, weights=PARAM["EL_WFN"][:,d], k=num_walkers)
        # Initialize Cavity Photon
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"][:] = random.choices(mylist, weights=PARAM["PHOT_WFN"][:], k=num_walkers)

        # Various observables
        PARAM["ENERGY_AVE"]        = np.zeros( (PARAM["num_steps_production"]) )
        PARAM["ENERGY_AVE_VAR"]    = np.zeros( (PARAM["num_steps_production"]) )
        PARAM["ENERGY_AVE_STD"]    = np.zeros( (PARAM["num_steps_production"]) )
        PARAM["ENERGY_TRIAL"]      = np.zeros( (PARAM["num_steps_production"]) )
        PARAM["N_WALKERS_CURRENT"] = num_walkers

    except KeyError:
        PARAM["POSITIONS"] = np.zeros( (particles,num_walkers,dimension) )
        for Ri,R in enumerate( R_NUC ):
            for dim in range(dimension):
                NSTART = Ri*(num_walkers//len(R_NUC))
                NEND   = (Ri+1)*(num_walkers//len(R_NUC))
                #PARAM["POSITIONS"][:,NSTART:NEND,dim]  = np.random.normal(R[dim], 2, size=(particles,NEND-NSTART))
                PARAM["POSITIONS"][:,NSTART:NEND,dim]  = (np.random.uniform(size=(particles,NEND-NSTART))*2-1)*10

        # Initialize Cavity Photon
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] = np.random.uniform(size=(num_walkers))*2-1 # Assuming only a single dimension/mode # TODO
            PARAM["QC"] *= 1/np.sqrt( PARAM["CAVITY_FREQ"] ) # Maximum QC for ground state of classical HO

        # Various observables
        PARAM["ENERGY_AVE"]        = np.zeros( (PARAM["num_steps_equilibration"]) )
        PARAM["ENERGY_AVE_VAR"]    = np.zeros( (PARAM["num_steps_equilibration"]) )
        PARAM["ENERGY_AVE_STD"]    = np.zeros( (PARAM["num_steps_equilibration"]) )
        PARAM["ENERGY_TRIAL"]      = np.zeros( (PARAM["num_steps_equilibration"]) )
        PARAM["N_WALKERS_CURRENT"] = np.zeros( (PARAM["num_steps_equilibration"]) )

    return PARAM

def updateDATA( PARAM ):

    # Various observables
    PARAM["ENERGY_AVE"]     = np.average( get_potential(PARAM) )
    PARAM["ENERGY_AVE_VAR"] = np.var( get_potential(PARAM) )
    PARAM["ENERGY_AVE_STD"] = np.std( get_potential(PARAM) )
    PARAM["ENERGY_TRIAL"]   = PARAM["E_TRIAL_GUESS"]
    PARAM["N_WALKERS_CURRENT"] = len( PARAM["POSITIONS"][:,0] )

    if ( PARAM["STEP_CURRENT"] % (num_steps//50) == 0 and PARAM["STEP_CURRENT"] != 0 ):
        print ( f"Step = {PARAM["STEP_CURRENT"]}, Nw = {PARAM["N_WALKERS_CURRENT"]}, <E> = {round(PARAM["ENERGY_AVE"],3)}" )

    return PARAM

def QMC( PARAM ):

    # Initialize QMC variables
    PARAM = prepare( PARAM )

    # Equilibration loop
    PARAM["EQUILIBRATION_FLAG"] = True
    for step in range(num_steps):
        PARAM["STEP_CURRENT"] = step
        PARAM = updateDATA( PARAM ) # Save <E>

        walkerLIST = np.arange( PARAM["N_WALKERS_CURRENT"] )
        with mp.Pool(processes=NCPUS) as pool:
            RESULT = pool.map(DMC_step, walkerLIST, PARAM)
            PARAM[]






    # Do equilibrium run
    positions, TRAJ, energy_traj, WFNs, PARAM = DMC(PARAM) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, WFNs, production_flag=False)

    # Do production run starting from equilibrated positions
    positions, TRAJ, energy_traj, WFNs, PARAM = DMC(PARAM,positions=positions) # This can be very easily parallelized.
    print_results( positions, energy_traj, PARAM )
    plot(positions, TRAJ, energy_traj, PARAM, WFNs, production_flag=True)

