import numpy as np

def Bruggeman_EMT(m,f):
    """
    Function that generate the optical index of a mixture of several components
    using the Bruggeman mixing rule
    Input:
        - m : array of optical indices of the components
        - f : array of volume fractions of the components
    Output:
        - m_eff:  optical indices of the mix
    For 2 components : find 2nd degree polynomial root
    For more than 2 components : iterative solution
    C. Pinte
    28/09/11
    27/10/11 : more than 2 components
    18/09/13 : changing iteration for a much more stable scheme
    14/12/18 : J. Milli porting to python
    """

    n_iter_max = 100
    accuracy = 1.e-6
    m = np.asarray(m,dtype=complex)
    f = np.asarray(f,dtype=float)
    n = len(m) # Number of components
    # Sanity check
    if (n  !=  len(f)):
        print('Incorrect number of components')
        raise IOError('Bruggeman rule error')
    eps = m**2 # Constantes dielectriques
        
    if (n==2) : # equation du degre 2
        b = 2*f[0]*eps[0] - f[0]*eps[1] + 2*f[1]*eps[1] -f[1]*eps[0]
        c = eps[0] * eps[1]
        a = -2.   # --> 1/2a  = -0.25
        delta = b*b - 4*a*c
        eps_eff1 = (-b - np.sqrt(delta)) * (-0.25)
        eps_eff2 = (-b + np.sqrt(delta)) * (-0.25)
        # Selection (et verif) de l'unique racine positive (a priori c'est eps_eff1)
        if eps_eff1.imag > 0:
            if eps_eff2.imag > 0:
                raise TypeError("Bruggeman EMT rule: all imaginary parts are > 0")
            eps_eff = eps_eff1
        else:
            if eps_eff2.imag < 0:
                raise TypeError('Bruggeman EMT rule: all imaginary parts are < 0')
            eps_eff = eps_eff2
    
    else: # n > 2, pas de solution analytique, methode iterative
        eps_eff = np.sum(f*eps**(1./3))**3 # Initial guess : Landau, Lifshitz, Looyenga rule
        # Marche aussi avec eps_eff = 1.0 mais necessite quelques iterations de plus
        eps_eff1 = eps_eff
        k=0 # in case the iteration does not work
        while k < n_iter_max:
            k=k+1
            S = np.sum((eps-eps_eff)/(eps+2*eps_eff)*f )
            eps_eff =  eps_eff * (2*S+1)/(1-S)
            if (np.abs(S) < accuracy):
                print('Accuracy reached at the {0:d}th iteration. Exiting'.format(k))
                break
    
        if (k==n_iter_max) :
            print("****************************************************")
            print("WARNING: Bruggeman not converging")
            print("Using Landau, Lifshitz, Looyenga instead")
            print("****************************************************")
            eps_eff = eps_eff1
     
    # Indices optiques
    m_eff = np.sqrt(eps_eff)
    # Verification finale ---> OK
    #S = sum( (eps(:)-eps_eff)/(eps(:)+2*eps_eff) * f(:) )
    #write(*,*) "Verif Bruggeman", lambda, iter, m_eff, abs(S)
    return m_eff


if __name__ == "__main__":
    index1 = 1.2+0.025*1j
    index2 = 1.41+0.01*1j
    index3 = 1.7+0.001*1j
    mix = Bruggeman_EMT([index1,index2,index3],[0.2,0.7,0.1])
    print(mix)