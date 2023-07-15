import numpy as np

def get_potential(x, PARAM):

    interacting     = PARAM["interacting"]
    R_NUC           = PARAM["R_NUC"]
    Z_NUC           = PARAM["Z_NUC"]


    if ( len(x.shape) == 3 ):
        NELECTRONS, NWALKERS, NDIMENSIONS  = x.shape
        V_EL = np.zeros( (NELECTRONS, NWALKERS) )

        # Electron-Electron Interaction
        if ( interacting == True ):
            for p1 in range( NELECTRONS ):
                for p2 in range( p1+1, NELECTRONS):
                    if ( p1 != p2 ):
                        Ree = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V_EL[p1,:] += 1 / Ree

        # Electron-Nuclear Interaction
        for p1 in range(NELECTRONS): # Loop over particles
            for Ri, R in enumerate( R_NUC ): # Loop over nuclei
                ReN = R_NUC[Ri,:] - x[p1,:,:] # Automatically subtracts over final dimension
                ReN = np.einsum( "wd->w", ReN**2 ) ** (1/2)
                V_EL[p1,:] -= Z_NUC[Ri] / ReN
        
        # Add all QM particles
        V_EL = np.einsum("pw->w", V_EL[:,:]) # Note: Shape Change

        # Nuclear-Nuclear Interaction
        V_NUC = 0
        if ( len(R_NUC) >= 2 ):
            for Ri1, R1 in enumerate( R_NUC ):
                for Ri2, R2 in enumerate( R_NUC ):
                    if ( Ri2 >= Ri1 + 1 ):
                        R12 = np.linalg.norm( R1 - R2 )
                        V_NUC += Z_NUC[Ri1] * Z_NUC[Ri2] / R12

        # Photonic Cavity Contributions
        if ( PARAM["DO_POLARITON"] == True ):
            WC     = PARAM["CAVITY_FREQ"] 
            A0     = PARAM["CAVITY_COUPLING"]
            EPOL   = PARAM["CAVITY_POLARIZATION"]
            QC     = PARAM["QC"] # QC is propagated in QMC.py

            ### Photonic Energy: 0.5 WC**2 QC**2 ###
            V_PH = 0.5 * WC**2 * QC[:]**2

            ### Direct Interaction ###
            # WC A0 (MU.E) QC CAVITY_POLARIZATION
            MU_NUC      = np.einsum("N,Nd->d", Z_NUC[:], R_NUC[:,:] ) # Sum over nuclei
            MU_EL       = np.einsum("ewd->wd", x[:,:,:] ) # Sum over electrons
            
            MU_TOT = MU_NUC[:] - MU_EL[:,:]

            MU_TOT_PROJ = np.einsum( "wd,d->w", MU_TOT, EPOL ) # Project along field polarization
            V_elph      = np.sqrt(2 * WC**3) * A0 * MU_TOT_PROJ[:] * QC[:] # QC = 1/sqrt(2WC) * ( a.T + a )


            """

            ### DIPOLE SELF-ENERGY ###
            # mu**2 = T1 + T2 + T3
            # T1 =    \sum_{p,p'}^{N_el} x_p * x_p'
            # T2 = -2*\sum_{p}^{N_el} \sum_{I}^{N_IONS} x_p * R_I
            # T3 =    \sum_{I,I'}^{N_IONS} R_I * R_I'
            # This might be same as squaring the mu from the previous direct interaction term
            #       I just wonder about the p != pp terms. ~BMW
            T1 = np.zeros( (NWALKERS) )
            for p in range( NELECTRONS ):
                for pp in range( NELECTRONS ):
                    # This has a plus sign since (-x)*(-x)=x^2
                    T1 += np.einsum( "wd,d->w", x[p,:,:] * x[pp,:,:], EPOL )
            
            T2 = np.zeros( (NWALKERS) )
            for p in range( NELECTRONS ):
                for Ri, R in enumerate( R_NUC ):
                    # This has a minus sign since (-x)*(R)=-xR
                    # Factor 2.000 comes from the binomial expansion
                    T2 += -2.000 * np.einsum( "wd,d->w", x[p,:,:] * Z_NUC[Ri] * R_NUC[Ri,:], EPOL )

            T3 = 0
            for Ri1, R1 in enumerate( R_NUC ):
                for Ri2, R2 in enumerate( R_NUC ):
                    # This has a plus sign since (R)*(R)=-R^2
                    T3 += np.einsum( "d,d->", Z_NUC[Ri1] * R_NUC[Ri1,:] * Z_NUC[Ri2] * R_NUC[Ri2,:], EPOL )


            MU_2 = T1 + T2 + T3
            V_DSE = WC * A0**2 * MU_2

            """
            V_DSE = WC * A0**2 * MU_TOT_PROJ**2



            return V_EL + V_NUC + V_PH + V_elph + V_DSE

        else:
            return V_EL + V_NUC
    
    else:
        dimension = len(R_NUC[0,:])
        V = np.zeros( (len(x), dimension) )
        for d in range( dimension ): # Dimension
            for Ri, R, in enumerate( R_NUC ):
                V[:,d] += -Z_NUC[Ri]/np.abs(x-R[d]) # Only Electron-Nuclei Interaction in dim = d
        return V