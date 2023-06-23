import numpy as np

from potential import get_potential

def DMC( PARAM, positions=None ):

    particles = PARAM["particles"]
    num_walkers = PARAM["num_walkers"]
    dimension = PARAM["dimension"]
    R_NUC = PARAM["R_NUC"]
    E_TRIAL = PARAM["E_TRIAL_GUESS"]

    if ( positions is None ):
        num_steps = PARAM["num_steps_equilibration"]
        time_step = PARAM["time_step_equilibration"]
        alpha     = PARAM["alpha_equilibration"]

        positions = np.zeros( (particles,num_walkers,dimension) )
        for Ri,R in enumerate( R_NUC ):
            for dim in range(dimension):
                NSTART = Ri*(num_walkers//len(R_NUC))
                NEND   = (Ri+1)*(num_walkers//len(R_NUC))
                #positions[:,NSTART:NEND,dim]  = np.random.normal(R[dim], 2, size=(particles,NEND-NSTART))
                positions[:,NSTART:NEND,dim]  = (np.random.uniform(size=(particles,NEND-NSTART))*2-1)*10

    else:
        positions = positions
        num_steps = PARAM["num_steps_production"]
        time_step = PARAM["time_step_production"]
        alpha     = PARAM["alpha_production"]


    trajectory  = np.zeros( (particles,100,dimension,num_steps) ) # Store first 100 walker for visualization
    energy_traj = np.zeros( (num_steps,3) )

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):

        energy_traj[step,0] = np.average( get_potential(positions,PARAM) )
        energy_traj[step,1] = np.var( get_potential(positions,PARAM) )
        energy_traj[step,2] = np.std( get_potential(positions,PARAM) )

        if ( step % (num_steps//10) == 0 ):
            print ( f"Step = {step}, N = {len(positions[0,:,0])}" )

        # Propagate the walkers
        positions += np.random.normal(0, np.sqrt(time_step), size=(particles,len(positions[0,:]),dimension))

        # Compute the potential energy for each walker
        potential_energies = get_potential(positions,PARAM)

        # Compute the weights of the walkers
        E_TRIAL  = E_TRIAL + alpha * np.log( num_walkers / len(positions[0,:]) )
        ##### TODO -- TEST. Does this work better ? ~ BMW #####
        #E_TRIAL  = energy_traj[step,0] + alpha * np.log( num_walkers / len(positions[0,:]) )
        
        dE       = potential_energies - E_TRIAL
        POW_THRESH = 7 # This is somewhat arbitrary
        if ( (-time_step * dE).any() > POW_THRESH ):
            INDICES = [ j for j,de in enumerate(dE) if (-time_step * de) > POW_THRESH ]
            print( "HERE:", len(INDICES) )
            dE[ INDICES ] = -7 / time_step
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
        s[ IND_2 ]     = 2 # Duplicate these

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

        if ( len(positions[0]) > 100 ):
            trajectory[:,:,:,step] = positions[:,:100,:]


        if ( step == 0 ):
            WFN, EDGES = np.histogram( positions[:,:].flatten(), bins=np.linspace(-10,10,1000) )
        else:
            TMP = np.histogram( positions[:,:].flatten(), bins=np.linspace(-10,10,1000) )
            WFN += TMP[0]

    return positions, trajectory, energy_traj, (EDGES, WFN)



