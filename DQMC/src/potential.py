import numpy as np

def get_potential(x, PARAM):

    interacting     = PARAM["interacting"]
    R_NUC           = PARAM["R_NUC"]
    Z_NUC           = PARAM["Z_NUC"]


    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V_EL = np.zeros( (shapes[0], shapes[1]) )

        # Electron-Electron Interaction
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range( p1+1, shapes[0]):
                    if ( p1 != p2 ):
                        Ree = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V_EL[p1,:] += 1 / Ree

        # Electron-Nuclear Interaction
        for p1 in range(shapes[0]): # Loop over particles
            for Ri, R, in enumerate( R_NUC ): # Loop over nuclei
                ReN = R_NUC[Ri,:] - x[p1,:,:] # Automatically subtracts over final dimension
                ReN = np.einsum( "wd->w", ReN**2 ) ** (1/2)
                V_EL[p1,:] -= Z_NUC[Ri] / ReN
        
        # Add all QM particles
        V_EL = np.einsum("pw->w", V_EL[:,:]) # Note: Shape Change

        # Nuclear-Nuclear Interaction
        V_NUC = 0
        if ( len(R_NUC) >= 2 ):
            for Ri1, R1, in enumerate( R_NUC ):
                for Ri2, R2, in enumerate( R_NUC ):
                    if ( Ri2 >= Ri1 + 1 ):
                        R12 = np.linalg.norm( R1 - R2 )
                        V_NUC += Z_NUC[Ri1] * Z_NUC[Ri2] / R12

        # Photonic Cavity Contributions
        if ( PARAM["DO_POLARITON"] == True ):
            WC     = PARAM["CAVITY_FREQ"] 
            A0     = PARAM["CAVITY_COUPLING"]
            EPOL   = PARAM["CAVITY_POLARIZATION"]
            QC     = PARAM["QC"] # QC is propagated in QMC.py

            # Photonic Energy: 0.5 WC**2 QC**2
            V_PH = 0.5 * WC**2 * QC[:]**2

            # Direct Interaction
            # WC A0 (MU.E) QC CAVITY_POLARIZATION
            MU_NUC      = np.einsum("N,Nd->d", Z_NUC[:], R_NUC[:,:] ) # Sum over nuclei 
            MU_EL       = np.einsum("ewd->wd", x[:,:,:] ) # Sum over electrons
            MU_TOT      = np.einsum("wd,wd->wd", MU_NUC[None,:], -1.0000 * MU_EL[:,:]) # (w,d)
            MU_TOT_PROJ = np.einsum( "wd,d->w", MU_TOT, EPOL ) # Project along field polarization
            V_elph      = WC * A0 * MU_TOT_PROJ

            # NEED TO CALCULATE DIPOLE SELF-ENERGY YET # TODO
            # How to write this term ? Need to be careful here.
            #V_DSE = WC * A0**2 * MU_TOT**2
            # V_DSE ~ d(1)*d(1) + d(1)*d(2)
            TMP  = np.einsum( "wd,d->w", x[0,:,:], EPOL ) * np.einsum( "wd,d->w", x[0,:,:], EPOL )
            TMP += np.einsum( "wd,d->w", x[0,:,:], EPOL ) * np.einsum( "wd,d->w", x[1,:,:], EPOL )
            TMP += np.einsum( "wd,d->w", x[1,:,:], EPOL ) * np.einsum( "wd,d->w", x[1,:,:], EPOL )
            
            V_DSE = WC * A0**2 * TMP
            #V_DSE = WC * A0**2 * np.einsum( "wd,d->w", np.einsum("Iwd->Jwd",  ), EPOL )

            return V_EL + V_NUC + V_PH + V_elph + V_DSE

        else:
            return V_EL + V_NUC
    
    else:
        V = x*0
        for Ri, R, in enumerate( R_NUC ):
            V += -Z_NUC[Ri]/np.abs(x-R[0]) # Only Electron-Nuclei Interaction in dim = 0
        return V