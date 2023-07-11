import numpy as np

from potential import get_potential

def DMC( PARAM, positions=None ):

    particles   = PARAM["particles"]
    num_walkers = PARAM["num_walkers"]
    dimension   = PARAM["dimension"]
    R_NUC       = PARAM["R_NUC"]
    interacting = PARAM["interacting"]

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
                positions[:,NSTART:NEND,dim]  = (np.random.uniform(size=(particles,NEND-NSTART))*2-1)*20

        # Initialize Cavity Photon
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] = np.random.uniform(size=(num_walkers))*2-1 # Assuming only a single dimension/mode # TODO

    else:

        # Use final configuration as initial guess
        positions = positions
        PARAM["QC"] = PARAM["QC"]


        # Sample from equilibrium wavefunction -- THIS GIVES ANOTHER (ALBEIT SHORTER) EQUILIBRIUM TIME, WHICH IS ANNOYING ~BMW
        #positions = np.zeros( (particles,num_walkers,dimension) )
        #DATA_DIR = "DATA_" + PARAM["DATA_DIR"]
        #for dim in range( dimension ):
        #    EDGES, EL_WFN = np.loadtxt(f"{DATA_DIR}/WAVEFUNCTION_d{dim}_N{particles}_INT_{interacting}_Equilibration.dat" ).T
        #    DX            = EDGES[1] - EDGES[0]
        #    PROB          = EL_WFN**2 * DX
        #    positions[:,:,dim]  = np.random.choice( EDGES, p=PROB, size=(particles,num_walkers) )
        #if ( PARAM["DO_POLARITON"] == True ):
        #    EDGES, PHOT_WFN = np.loadtxt(f"{DATA_DIR}/PHOTON_WAVEFUNCTION_d{dim}_N{particles}_INT_{interacting}_Equilibration.dat" ).T
        #    DX            = EDGES[1] - EDGES[0]
        #    PROB          = PHOT_WFN**2 * DX
        #    PARAM["QC"]  = np.random.choice( EDGES, p=PROB, size=(num_walkers) )
        
        
        num_steps = PARAM["num_steps_production"]
        time_step = PARAM["time_step_production"]
        alpha     = PARAM["alpha_production"]


    trajectory  = np.zeros( (particles,100,dimension,num_steps) ) # Store first 100 walker for visualization
    energy_traj = np.zeros( (num_steps,5) )
    if ( PARAM["DO_POLARITON"] == True ):
        PARAM["QC_TRAJ"] = np.zeros( (100,num_steps) ) # Store first 100 walker for visualization

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):
        positions_old = positions
        potential_energies_old = get_potential(positions_old,PARAM)

        energy_traj[step,0] = np.average( get_potential(positions,PARAM) )
        energy_traj[step,1] = np.var( get_potential(positions,PARAM) )
        energy_traj[step,2] = np.std( get_potential(positions,PARAM) )
        energy_traj[step,3] = PARAM["E_TRIAL_GUESS"]
        energy_traj[step,4] = len(positions[0,:]) # Current number of walkers

        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC_TRAJ"][:,step] = PARAM["QC"][:100]

        if ( step % (num_steps//20) == 0 ):
            print ( f"Step = {step}, N = {len(positions[0,:,0])}, E_T = {round(energy_traj[step,3],3)}, <E> = {round(energy_traj[step,0],3)}" )

        # Propagate the walkers
        positions += np.random.normal(0, np.sqrt(time_step), size=(particles,len(positions[0,:]),dimension))
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] += np.random.normal( 0, np.sqrt(time_step), size=( len(positions[0,:]) ) )

        # Compute the potential energy for each walker
        potential_energies     = get_potential(positions,PARAM)
        AVE_Pot                = (potential_energies+potential_energies_old)/2

        # Compute the weights of the walkers
        PARAM["E_TRIAL_GUESS"] = np.average( potential_energies_old ) + alpha * np.log( num_walkers / len(positions[0,:]) )

        #dE = potential_energies - PARAM["E_TRIAL_GUESS"]
        dE = AVE_Pot - PARAM["E_TRIAL_GUESS"]

        # POW_THRESH = 7 # This is somewhat arbitrary
        # if ( (-time_step * dE).any() > POW_THRESH ):
        #     INDICES = [ j for j,de in enumerate(dE) if (-time_step * de) > POW_THRESH ]
        #     dE[ INDICES ] = -POW_THRESH / time_step
        # elif ( (-time_step * dE).any() < -POW_THRESH ):
        #     dE[ INDICES ] = POW_THRESH / time_step

        weights  = np.exp( -time_step * dE )

        # Evaluate q = s + r
        RAND   = np.random.uniform( size=len(positions[0,:]) )
        s      = np.floor( weights + RAND ).astype(int)
        s[s<0] = 0 # I found some issues without this.
        s[s>3] = 2 # The number of walkers can go crazy.


        """
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
        """

        if ( np.sum( s ) < 1 ):
            print( "WARNING !!!!" )
            print( "Number of walkers went to zero..." )
            #print( "Skip branching...Probably need to increase number of walkers" )
            #continue
            exit()
        #if ( (s < 0).any() ):
        #    print( s[s < 0] )
        #    exit()


        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP)
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] = np.repeat( PARAM["QC"], s[:] )
        """
        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP).reshape( (particles,np.sum(s),dimension) )
        if ( PARAM["DO_POLARITON"] == True ):
            PARAM["QC"] = np.repeat( PARAM["QC"], s[:] )
        """

        if ( len(positions[0]) > 100 ):
            trajectory[:,:,:,step] = positions[:,:100,:]

        if ( step == 0 ):
            #NEDGES = 1000
            dEDGE = 0.01
            XMIN   = -50 #np.min( R_NUC[:,:]*5 )
            XMAX   =  50  #np.max( R_NUC[:,:]*5 )
            #XMAX   = np.max( [abs(XMIN),abs(XMAX)] )
            #XMIN   = -XMAX
            EDGES = np.arange(XMIN,XMAX+dEDGE,dEDGE)
            NEDGES = len(EDGES)
            EL_WFN = np.zeros( ( NEDGES-1, dimension ) )
            for d in range( dimension ):
                EL_WFN[:,d], EDGES = np.histogram( positions[:,:,d].flatten(), bins=EDGES ) #bins=np.linspace(XMIN,XMAX,NEDGES+1)
            if ( PARAM["DO_POLARITON"] == True ):
                PHOT_WFN, _ = np.histogram( PARAM["QC"], bins=EDGES )
        else:
            for d in range( dimension ):
                TMP     = np.histogram( positions[:,:,d].flatten(), bins=EDGES )
                EL_WFN[:,d] += TMP[0]
            if ( PARAM["DO_POLARITON"] == True ):
                TMP = np.histogram( PARAM["QC"], bins=EDGES )
                PHOT_WFN += TMP[0]


    if ( PARAM["DO_POLARITON"] == True ):
        return positions, trajectory, energy_traj, (EDGES, EL_WFN, PHOT_WFN), PARAM
    else:
        return positions, trajectory, energy_traj, (EDGES, EL_WFN), PARAM



