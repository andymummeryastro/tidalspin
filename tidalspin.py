import numpy as np 
from numpy import cos, sin, pi, roots
from astropy import constants 
from tqdm import tqdm
import matplotlib.pyplot as plt

Ms = constants.M_sun.value
Rs = constants.R_sun.value
c = constants.c.value
G = constants.G.value

settup_str = """
  ________________
 |_____    _______| ____    _______   __        _______  ______  
       |  |    (_) |  _ \  /  __   \ |  |      |   ____||   (_)|(_)  ______
       |  |    | | | | | ||  |__|  | |  |      |  |___  |   ___|| | |  __  |
       |  |    | | | |_| ||  |  |  | |  |____  |___   | |  |    | | | |  | |
       |__|    |_| |____/ |__|  |__| |_______|  ___|  | |__|    |_| |_|  |_|
                                               |______|

             TidalSpin was created by Andrew Mummery*

              * andrew.mummery@physics.ox.ac.uk
"""

print(settup_str)## Forgive me, I coded a lot of this during a long train journey. 

def main():

    format_plots_nicely()

    ### Produce 2D mass spin contours with Monte Carlo, and two 1D posteriors. 
    logM = 8.5##Some prior log10(mass) estimate. 
    sigmaM = 0.3## uncertainty in log_10(black hole mass) from M-sigma relationship

    prior_MBH = lambda lm: log_norm(lm, logM, sigmaM) * (10**lm/1e8)**0.03 * np.exp(-(10**lm/(6.4e7))**0.49)## Assume log-normal mass prior. 

    hm, mass_matrix = get_hills_masses()## Get Hills masses for solar mass star (for comparison)

    p_a, a = one_d_spin_dist(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10)## Get 1D spin posterior 
    p_m, m = one_d_mass_dist(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10)## Get 1D mass posterior
    a_samples, m_samples, _, _ = monte_carlo_all(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10, N_draw=100000)
    ## Monte Carlo 100,000 TDEs to verify 1D posteriors and determine 2D correlations. 

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax2.axis('off')

    ax3.scatter(m_samples, a_samples, marker='.', color='blue', alpha=0.02, rasterized=True)
    ax3.set_xlabel(r'$\log_{10}M_\bullet/M_\odot$', fontsize=30)
    ax3.set_ylabel(r'$a_\bullet$', fontsize=30)
    ax3.plot(np.log10(hm/Ms), a, ls='-.', c='k')
    ax3.set_ylim(-1, 1)
    ax3.set_xlim(right=9.2)
    ax3_xlim = ax3.get_xlim()


    ax1.hist(m_samples, bins=20, density=True, color='blue')
    ax1.plot(m, p_m, c='r', ls='--')
    ax1.set_ylabel(r'$p(\log_{10} M_\bullet/M_\odot)$', fontsize=30)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_xlim(ax3_xlim)

    ax4.hist(a_samples, bins=30, density=True, color='blue')
    ax4.plot(a, p_a, ls='--', c='r')

    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r'$p(a_\bullet)$', fontsize=30, rotation=270, labelpad=40)
    ax4.set_xlabel(r'$a_\bullet$', fontsize=30)
    ax4.set_xlim(-1, 1)
    ax4.yaxis.tick_right()
    ax4.set_yticks([])
    ax4.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()



def format_plots_nicely():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'
    return 

    
def get_ibso(a, psi):
    cs = [a**8 * cos(psi)**4, 
          0, 
          -2 * a**6 * cos(psi)**2 + 6 * a**6 * cos(psi)**4,
          -8 * a**4 * cos(psi)**2 + 16 * a**4 * cos(psi)**2 * sin(psi)**2, 
          a**4 - 4 * a**4 * cos(psi)**2 + 9 * a**4 * cos(psi)**4,  
          8 * a**2 - 24 * a**2 * cos(psi)**2 - 16 * a**2 * sin(psi)**2,  
          16 - 2 * a**2 + 6 * a**2 * cos(psi)**2, 
          -8, 
          1]
    
    root = roots(cs[::-1])

    if a > 1:
        return np.nan

    real_roots = root[np.isreal(root)]
    if len(real_roots)!=2:
        if a > 0:
            return np.real(real_roots[1])
        return np.real(real_roots[0])
        
    if a > 0:
        return np.real(min(real_roots))
    return np.real(max(real_roots))


def hills_mass(a, psi, eta=1, mstar=1, rstar=1):
    x = get_ibso(a, psi)
    amp = (2/eta * c**6 * (rstar * Rs)**3/(G**3 * mstar * Ms))**0.5 * 1/x**1.5 
    fac = (1 + 6*x/(x**2 - a**2*cos(psi)**2) + 3*a**2/(2*x**2) - 6*a*sin(psi)/x**1.5 * (x**2/(x**2 - a**2*cos(psi)**2))**0.5)**0.5
    return amp * fac


def get_hills_masses(N_spin = 300, 
                  N_psi = 200):


    psi_ = np.linspace(0.001, pi/2, N_psi)
    a_ = np.linspace(-0.9999, +0.9999, N_spin)

    print('Generating Hills masses......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)

    max_m = np.zeros_like(a_)
    for j in range(len(a_)):
        max_m[j] = max(max_mass_matrix[j, :])

    return max_m, max_mass_matrix
    

#### Sampling many objects
def MassRadiusRelation(mstar):
    rstar = np.zeros_like(mstar)
    i_low = mstar < Ms 
    i_high = ~i_low 
    rstar[i_low] = Rs * np.power(mstar[i_low]/Ms,0.56)
    rstar[i_high] = Rs * np.power(mstar[i_high]/Ms,0.79)
    return rstar

def KroupaIMF(mstar):
    i_lowest = mstar < 0.08 * Ms 
    i_low = (mstar < 0.5 * Ms) * (~ i_lowest)
    i_med = (mstar < 1 * Ms) * (~ i_lowest) *  (~ i_low)
    i_high = (~ i_lowest) * (~ i_low) * (~i_med)

    P = np.zeros_like(mstar)

    P[i_lowest] = (mstar[i_lowest]/(0.08*Ms))**(-0.3)
    P[i_low] = (mstar[i_low]/(0.08*Ms))**(-1.8)
    P[i_med] = (mstar[i_med]/(0.08*Ms))**(-2.7) * (0.5/0.08)**(-1.8+2.7)
    P[i_high] = (mstar[i_high]/(0.08*Ms))**(-2.3)  * (1/0.08)**(-2.7+2.3) * (0.5/0.08)**(-1.8+2.7)

    return P
    
def TDE_rate_on_stellar_params(mstar):
    rstar = MassRadiusRelation(mstar)
    return np.power(mstar/Ms,-1/3) * np.power(rstar/Rs, 1/4)

def StellarMassDistributionFunction(Mmin=0.01*Ms, Mmax=2.0*Ms, N=10000):
    mstars = np.logspace(np.log10(Mmin), np.log10(Mmax), N)
    dM = (mstars[1:] - mstars[:-1])
    ps = KroupaIMF(mstars) * TDE_rate_on_stellar_params(mstars)
    return ps[:-1]/np.sum(ps[:-1] * dM)


### Log-normal for black hole masses 
def log_norm(x, log_mu, log_sigma):
    ### Mean must be in log10, as must x. 
    return 1/(np.sqrt(2*np.pi)*log_sigma) * np.exp(-(x-log_mu)**2/(2*log_sigma**2))

def one_d_spin_fixed_mass(Mbh, 
                  prior_spin=None, 
                  max_mass_matrix=None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_spin = 300, 
                  N_psi = 200, 

                  eta = 1):
    
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms

    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

    dpsi = psi_[1] - psi_[0]
    da_ = a_[1] - a_[0]

    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    p_a = np.zeros(N_spin)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)


    for i, a in enumerate(a_):
        p = 0
        p_ = np.heaviside(mass_mass_matrix[i] - Mbh, 1) 
        p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_spin(a) * da_
        
        p_a[i] = p

    p_a = p_a
    p_a = p_a/sum(p_a * da_)
    
    return p_a, a_




def one_d_spin_dist(prior_Mbh, log_Mbh_min, log_Mbh_max, 
                  prior_spin=None,
                  max_mass_matrix = None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1, 
                  
                  likelihood=None): 
    
    """
    Inputs: 
        prior_Mbh = function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max = extent of black hole mass range to compute posterior over. 
        prior_spin = function that accepts dimensionless spin and returns a prior. 
        M_star_min, M_star_max = lower and upper bounds on stellar population. 


        prior_star, prior_psi = other possible priors, default to sensible ones if left blank. 
        N_X (etc.) the number of grid spaces for variable X. 

        eta = stellar parameter (see paper).  

        likelihood = a function that accepts M_bh/M_hills and returns a likelihood. Defaults to 1 if unspecified (as in paoer). 
    
    Returns: 
        a = list of spins the posterior is computed at 
        p_a = the values of the posterior

    Notes:
        Returned posterior p_a includes both prograde and retrograde spins.  
        If the absolute value of the spin is required use p(|a|) = p(a) + p(-a), 
        or in Python:

        abs_a = a[N_spin/2:]

        p_abs_a = p_a[N_spin/2:] + p_a[N_spin/2 - 1 :: -1]

        Note that this requires N_spin to be an even number. 

    """

    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]

        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    p_a = np.zeros(len(a_))

    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    if likelihood is None:
        likelihood = lambda x: np.ones_like(x)

    print('Getting 1D spin distribution.....')
    for i, a in enumerate(tqdm(a_)):
        p = 0
        for l_m_bh in log_Ms:
            m_bh = 10**l_m_bh * Ms
            p_ = np.heaviside(mass_mass_matrix[i] - m_bh, 1)  * likelihood(m_bh/mass_mass_matrix[i]) 
            p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_Mbh(l_m_bh) * dMbh * prior_spin(a) * da_
        
        p_a[i] = p

    p_a = p_a/sum(p_a * da_)
    
    return p_a, a_



def one_d_mass_dist(prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix = None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1,
                  
                  likelihood=None): 

    """
    Inputs: 
        prior_Mbh = function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max = extent of black hole mass range to compute posterior over. 
        prior_spin = function that accepts dimensionless spin and returns a prior. 
        M_star_min, M_star_max = lower and upper bounds on stellar population. 


        prior_star, prior_psi = other possible priors, default to sensible ones if left blank. 
        N_X (etc.) the number of grid spaces for variable X. 

        eta = stellar parameter (see paper).  

        likelihood = a function that accepts M_bh/M_hills and returns a likelihood. Defaults to 1 if unspecified (as in paoer). 
    
    Returns: 
        log_mbh = list of og10 black hole masses the posterior is computed at 
        p_mbh = the values of the posterior.

    """
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]


        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)

    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]

    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    if likelihood is None:
        likelihood = lambda x: np.ones_like(x)
    

    p_bh = np.zeros(len(log_Ms))

    print('Getting 1D mass distribution.....')
    for i, l_m_bh in enumerate(tqdm(log_Ms)):
        p = 0
        m_bh = 10**l_m_bh * Ms
        for j, a in enumerate(a_):
            p_ = np.heaviside(mass_mass_matrix[j] - m_bh, 1) * likelihood(m_bh/mass_mass_matrix[i]) 
            p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_Mbh(l_m_bh) * dMbh * prior_spin(a) * da_
    
        p_bh[i] = p

    p_bh = p_bh/sum(p_bh * dMbh)
    
    return p_bh, log_Ms


def monte_carlo_all(prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix=None,
                  prior_star=None, 
                  prior_psi=None,
                  
                  N_draw=10000,

                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 

                  Mstar_min=0.1, 
                  Mstar_max=1,
                  
                  eta = 1,
                  
                  likelihood=None):
    """
    Inputs: 
        prior_Mbh = function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max = extent of black hole mass range to compute posterior over. 
        prior_spin = function that accepts dimensionless spin and returns a prior. 
        M_star_min, M_star_max = lower and upper bounds on stellar population. 

        N_draw = number of Monte Carlo samples to perform. 

        prior_star, prior_psi = other possible priors, default to sensible ones if left blank. 
        N_X (etc.) the number of grid spaces for variable X. 

        eta = stellar parameter (see paper).  

        likelihood = a function that accepts M_bh/M_hills and returns a likelihood. Defaults to 1 if unspecified (as in paoer). 
    
    Returns: 
        a_samples, m_samples, mstar_samples, prior_samples = list of length N_draw of all the parameters from successful TDE samples. 

    Notes:
        Returned samples a_samples includes both prograde and retrograde spins.  

    """
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 
    
    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    PHI_MBH = np.cumsum(prior_Mbh(log_Ms)*dMbh)/np.sum(prior_Mbh(log_Ms)*dMbh)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]
        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]
        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)

    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    PHI_psis = np.cumsum(ppsi * dpsi)
    
    PHI_A = np.cumsum(prior_spin(a_)*da_)/np.sum(prior_spin(a_)*da_)


    m_samples = np.full(N_draw, np.nan)
    a_samples = np.full(N_draw, np.nan)
    m_star_samples = np.full(N_draw, np.nan)
    psi_samples = np.full(N_draw, np.nan)


    if likelihood is None:
        likelihood = lambda x: np.ones_like(x)

    print('Generating samples......')
    n=0

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        v = np.random.uniform(0, 1, 1)
        psi = psi_[np.argmin(abs(PHI_psis - v))]
        j = np.argmin(abs(psi - psi_))

        x = np.random.uniform(0, 1, 1)
        a_bh = a_[np.argmin(abs(PHI_A - x))]
        i = np.argmin(abs(a_ - a_bh))

        m_test = (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5 * eta**-0.5)

        p_ = (max_mass_matrix[i, j] > m_test) *  likelihood(m_test/max_mass_matrix[i, j]) 

        y = np.random.uniform(0, 1, 1)

        if p_>y:
            m_samples[n] = l_m_bh 
            a_samples[n] = a_bh
            m_star_samples[n] = m_star
            psi_samples[n] = psi
            n+=1
            pbar.update()


    a_samples = a_samples[a_samples==a_samples]
    m_samples = m_samples[m_samples==m_samples]
    psi_samples = psi_samples[psi_samples==psi_samples]
    m_star_samples = m_star_samples[m_star_samples==m_star_samples]

    return a_samples, m_samples, m_star_samples, psi_samples


def one_d_observable_dist(observe_dist, observe_values,
                  prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix = None,
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1): 
    
    p_tde, m_tde = one_d_mass_dist(max_mass_matrix=max_mass_matrix, prior_Mbh=prior_Mbh, prior_spin=prior_spin, prior_star=prior_star, prior_psi=prior_psi,
                                   log_Mbh_min=log_Mbh_min, log_Mbh_max=log_Mbh_max, N_star=N_star, Mstar_min=Mstar_min, 
                                   Mstar_max=Mstar_max, N_bh = N_bh, N_spin=N_spin, N_psi=N_psi, eta=eta)## Get 1D mass posterior
    

    dM = m_tde[1] - m_tde[0]

    p_observe = np.zeros_like(observe_values)

    for k, o in enumerate(observe_values):
        p_observe[k] = sum( observe_dist(o, np.array(m_tde)) * p_tde * dM )
    
    return p_observe


def monte_carlo_observable(observe_func, 
                  prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix=None,
                  prior_star=None, 
                  prior_psi=None,
                  
                  N_draw=10000,

                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 

                  Mstar_min=0.1, 
                  Mstar_max=1,
                  
                  eta = 1):
    
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 
    
    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    PHI_MBH = np.cumsum(prior_Mbh(log_Ms)*dMbh)/np.sum(prior_Mbh(log_Ms)*dMbh)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]    
        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    PHI_psis = np.cumsum(ppsi * dpsi)
    PHI_A = np.cumsum(prior_spin(a_)*da_)/np.sum(prior_spin(a_)*da_)


    m_samples = np.full(N_draw, np.nan)
    a_samples = np.full(N_draw, np.nan)
    m_star_samples = np.full(N_draw, np.nan)
    psi_samples = np.full(N_draw, np.nan)
    obs_samples = np.full(N_draw, np.nan)

    print('Generating samples......')
    n=0

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        v = np.random.uniform(0, 1, 1)
        psi = psi_[np.argmin(abs(PHI_psis - v))]
        j = np.argmin(abs(psi - psi_))

        x = np.random.uniform(0, 1, 1)
        a_bh = a_[np.argmin(abs(PHI_A - x))]
        i = np.argmin(abs(a_ - a_bh))

        p_ = max_mass_matrix[i, j] > (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5 * eta**-0.5)

        if p_:
            m_samples[n] = l_m_bh 
            a_samples[n] = a_bh
            m_star_samples[n] = m_star
            psi_samples[n] = psi
            obs_samples[n] = observe_func(l_m_bh)

            n+=1
            pbar.update()


    a_samples = a_samples[a_samples==a_samples]
    m_samples = m_samples[m_samples==m_samples]
    psi_samples = psi_samples[psi_samples==psi_samples]
    m_star_samples = m_star_samples[m_star_samples==m_star_samples]
    obs_samples = obs_samples[obs_samples==obs_samples]

    return obs_samples, a_samples, m_samples, m_star_samples, psi_samples






if __name__ == "__main__":
    main()