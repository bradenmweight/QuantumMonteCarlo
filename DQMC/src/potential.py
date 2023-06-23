import numpy as np

def get_potential(x, PARAM):

    interacting = PARAM["interacting"]
    R_NUC = PARAM["R_NUC"]
    Z_NUC = PARAM["Z_NUC"]



    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V = np.zeros( (shapes[0], shapes[1]) )

        # Electron-Electron Interaction
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range( p1+1, shapes[0]):
                    if ( p1 != p2 ):
                        Ree = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V[p1,:] += 1 / Ree

        # Electron-Nuclear Interaction
        for p1 in range(shapes[0]): # Loop over particles
            for Ri, R, in enumerate( R_NUC ): # Loop over nuclei
                ReN = R_NUC[Ri,:] - x[p1,:,:]
                ReN = np.einsum( "wd->w", ReN**2 ) ** (1/2)
                V[p1,:] -= Z_NUC[Ri] / ReN
        
        # Add all QM particles
        V = np.einsum("pw->w", V[:,:]) # Note: Shape Change

        # Nuclear-Nuclear Interaction
        V_NUC = 0
        if ( len(R_NUC) >= 2 ):
            for Ri1, R1, in enumerate( R_NUC ):
                for Ri2, R2, in enumerate( R_NUC ):
                    if ( Ri2 >= Ri1 + 1 ):
                        R12 = np.linalg.norm( R1 - R2 )
                        V_NUC += Z_NUC[Ri1] * Z_NUC[Ri2] / R12

        return V + V_NUC
    else:
        V = x*0
        for Ri, R, in enumerate( R_NUC ):
            V += -Z_NUC[Ri]/np.abs(x-R[0]) # Only Electron-Nuclei Interaction in dim = 0
        return V