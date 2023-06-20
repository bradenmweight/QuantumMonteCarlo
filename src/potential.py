import numpy as np

def get_potential(x, PARAM):

    interacting = PARAM["interacting"]
    R_NUC = PARAM["R_NUC"]
    Z_NUC = PARAM["Z_NUC"]



    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V = np.zeros( (shapes[0], shapes[1]) )

        # Electron-Electron Correlation
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range(shapes[0]):
                    if ( p1 != p2 ):
                        Ree = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V[p1,:] += 1/np.abs(Ree)

        # Electron-Nuclear Correlation
        for p1 in range(shapes[0]): # Loop over particles
            for Ni, R, in enumerate( R_NUC ):
                ReN = R_NUC[Ni,:] - x[p1,:,:]
                ReN = np.einsum( "wd->w", ReN**2 ) ** (1/2)
                V[p1,:] -= 1/np.abs(ReN)
        
        # Add all QM particles
        V = np.einsum("pw->w", V[:,:]) # Note: Shape Change

        # Nuclear-Nuclear Interaction
        V_NUC = 0
        if ( len(R_NUC) >= 2 ):
            for Ri1, R1, in enumerate( R_NUC ):
                for Ri2, R2, in enumerate( R_NUC ):
                    if ( Ri1 != Ri2 ):
                        R12 = np.linalg.norm( R1 - R2 )
                        V_NUC += Z_NUC[Ri1]*Z_NUC[Ri2]/R12

        return V + V_NUC
    else:
        V   = -0.001/np.abs(x) - 0.001/np.abs(x-1.65) # Only Electron-Nuclei Interaction
        return V