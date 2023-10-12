import numpy as np 
from numpy import cos, sin, pi, roots
from astropy import constants 
from tqdm import tqdm

settup_str = """
  ________________
 |_____    _______| ____    _______   __        _______  ______  
       |  |    (_) |  _ \  /  __   \ |  |      |   ____||   (_)|(_)  ______
       |  |    | | | | | ||  |__|  | |  |      |  |___  |   ___|| | |  __  |
       |  |    | | | |_| ||  |  |  | |  |____  |___   | |  |    | | | |  | |
       |__|    |_| |____/ |__|  |__| |_______|  ___|  | |__|    |_| |_|  |_|
                                               |______|

             TidalSpin was created by Andrew Mummery*
                It is still in development. 

              * andrew.mummery@physics.ox.ac.uk
"""

print(settup_str)## Forgive me, I coded this during a boring talk. 


def main():
    import matplotlib.pyplot as plt
    import matplotlib

    cmap = matplotlib.cm.get_cmap('coolwarm')

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


    if 1:
        logM = 8.75
        sigmaM = 0.3 

        p_a, a = get_spin_dist(logM=logM, sigmaM=sigmaM)
        
        prior_MBH = lambda m: log_norm(np.log10(m/Ms), logM, sigmaM)
        prior_spins = lambda a: np.ones_like(a)

        p_m, m = one_d_mass_dist(prior_Mbh=prior_MBH, M_bh_min=10**7, M_bh_max=10**9, prior_spin=prior_spins)

        hm = get_hills_masses()

        a_samples, m_samples, m_star_samples, psi_samples = monte_carlo_all(logM, sigmaM, N_draw=100000)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


        ax2.axis('off')

        ax3.scatter(m_samples, a_samples, marker='.', color='blue', alpha=0.02, rasterized=True)
        ax3.set_xlabel(r'$\log_{10}M_\bullet/M_\odot$', fontsize=30)
        ax3.set_ylabel(r'$a_\bullet$', fontsize=30)
        ax3.plot(np.log10(hm/Ms), a, ls='--', c='r')
        ax3.set_ylim(0, 1)


        ax1.hist(m_samples, bins=30, density=True, color='blue')
        ax1.plot(m, p_m, c='r', ls='--')
        ax1.set_ylabel(r'$p(\log_{10} M_\bullet/M_\odot)$', fontsize=30)
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_xticks([])
        ax1.set_xticklabels([])

        ax4.hist(a_samples, bins=30, density=True, color='blue')
        ax4.plot(a, p_a, ls='--', c='r')

        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel(r'$p(a_\bullet)$', fontsize=30, rotation=270, labelpad=40)
        ax4.set_xlabel(r'$a_\bullet$', fontsize=30)
        ax4.set_xlim(0, 1)
        ax4.yaxis.tick_right()
        ax4.set_yticks([])
        ax4.set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)

    if 0:
        def SMBHMassDistributionFunction(mBHs, Mc=1e6*Ms, alpha=0):
            m = mBHs/(1e6*Ms)
            ps = m**(alpha)/(1 + (Mc/mBHs)**(2-alpha)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
            return ps[:-1]/np.sum(ps[:-1] * (mBHs[1:] - mBHs[:-1]))

        N_bh = 10000
        mbhs = np.linspace(5, 10, N_bh) 
        p_no_tde = SMBHMassDistributionFunction(10**mbhs*Ms) * Ms * 10**np.mean(mbhs)
        N_draw = 100000
        
        m_tde = tde_mass_dist(N_draw=N_draw, black_hole_mass_dist=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, bh_spin=0) 
        ptde, _ = np.histogram(m_tde, bins=50, density=True)
        plt.hist(m_tde, bins=100, density=True, histtype=u'step')
        
        m_tde = tde_mass_dist(N_draw=N_draw, black_hole_mass_dist=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, bh_spin=0.5) 
        ptde, _ = np.histogram(m_tde, bins=50, density=True)
        plt.hist(m_tde, bins=100, density=True, histtype=u'step')


        m_tde = tde_mass_dist(N_draw=N_draw, black_hole_mass_dist=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, bh_spin=0.75) 
        ptde, _ = np.histogram(m_tde, bins=50, density=True)
        plt.hist(m_tde, bins=100, density=True, histtype=u'step')

        m_tde = tde_mass_dist(N_draw=N_draw, black_hole_mass_dist=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, bh_spin=0.9) 
        ptde, _ = np.histogram(m_tde, bins=50, density=True)
        plt.hist(m_tde, bins=100, density=True, histtype=u'step')

        m_tde = tde_mass_dist(N_draw=N_draw, black_hole_mass_dist=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, bh_spin=0.999) 
        ptde, _ = np.histogram(m_tde, bins=50, density=True)
        plt.hist(m_tde, bins=100, density=True, histtype=u'step')

        plt.yscale('log')


        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0, N_star=300) 
        plt.semilogy(m_tde, p_tde)
        
        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.5, N_star=300) 
        plt.semilogy(m_tde, p_tde)
        
        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.75, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.9, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.999, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        # plt.plot(mbhs[:-1], p_no_tde * ptde[0]/max(p_no_tde))

    if 0:
        def SMBHMassDistributionFunction(mBHs, Mc=5e5*Ms, alpha=-0.5):
            m = mBHs/(1e6*Ms)
            ps = m**(alpha)/(1 + (Mc/mBHs)**(2-alpha)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
            return ps[:-1]/np.sum(ps[:-1] * (mBHs[1:] - mBHs[:-1]))

        N_bh = 10000
        mbhs = np.linspace(5, 10, N_bh) 
        p_no_tde = SMBHMassDistributionFunction(10**mbhs*Ms)

        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0, N_star=300) 
        plt.semilogy(m_tde, p_tde)
        
        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.5, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.75, N_star=300) 
        plt.semilogy(m_tde, p_tde)
        
        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.9, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        p_tde, m_tde = one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, M_bh_max=1e10, M_bh_min=1e5, N_bh=N_bh, spin_bh=0.999, N_star=300) 
        plt.semilogy(m_tde, p_tde)

        plt.semilogy(mbhs[:-1], p_no_tde / np.sum(p_no_tde * (mbhs[1]-mbhs[0])), '--', c='k')

        plt.ylim(1e-5)

    plt.show()





Ms = constants.M_sun.value
Rs = constants.R_sun.value
c = constants.c.value
G = constants.G.value

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
    amp = (2/eta * c**6 * Rs**3/(G**3 * Ms))**0.5 * 1/x**1.5 
    fac = (1 + 6*x/(x**2 - a**2*cos(psi)**2) + 3*a**2/(2*x**2) - 6*a*sin(psi)/x**1.5 * (x**2/(x**2 - a**2*cos(psi)**2))**0.5)**0.5
    return amp * fac


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


### Log-normal 
def log_norm(x, log_mu, log_sigma):
    if log_sigma is None:
        return 1
    return 1/(np.sqrt(2*np.pi)*log_sigma) * np.exp(-(x-log_mu)**2/(2*log_sigma**2))

### Galactic scaling relationships 
def MH_Mgal(Mgal):
    return 10**(7.43 + 1.61 * np.log10(Mgal/3e10))

def MH_sig(sig):
    return 10**(7.87 + 4.384 * np.log10(sig/160))


def get_spin_dist(logM, sigmaM=None, 
                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  Mstar_min=0.1, 
                  Mstar_max=10):


    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]


    psi_ = np.linspace(0.001, pi/2, N_psi)
    dpsi = psi_[1] - psi_[0]


    a_ = np.linspace(-0.9999, +0.9999, N_spin)
    da_ = a_[1] - a_[0]

    print('Generating mass matrix......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)


    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    p_a = np.zeros(len(a_))

    if sigmaM is None:
        log_Ms = [logM]
        dMbh = 1
    else:
        log_Ms = np.linspace(logM - 5*sigmaM, logM + 5*sigmaM, N_bh)
        dMbh = (log_Ms[1] - log_Ms[0])

    print('Getting 1D spin distribution.....')
    for i, a in enumerate(tqdm(a_)):
        p = 0
        for l_m_bh in log_Ms:
            m_bh = 10**l_m_bh * Ms
            p_ = np.heaviside(mass_mass_matrix[i] - m_bh, 1)
            p += np.sum(np.sum(p_ * psd * dMs, axis=1) * sin(psi_) * dpsi, axis=0) * log_norm(l_m_bh, logM, sigmaM) * dMbh
        
        p_a[i] = p

    p_a = p_a/sum(p_a * da_)
    
    return p_a, a_


def get_hills_masses(N_spin = 300, 
                  N_psi = 200, 
                  Mstar=1):


    psi_ = np.linspace(0.001, pi/2, N_psi)
    a_ = np.linspace(-0.9999, +0.9999, N_spin)
    Mstar *= Ms

    print('Generating mass matrix......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)

    max_m = np.zeros_like(a_)
    for j in range(len(a_)):
        max_m[j] = max(max_mass_matrix[j, :])

    f = ((MassRadiusRelation(Mstar)/Rs)**1.5 * (Ms/Mstar)**0.5)
    max_m *= f 
    
    return max_m
    


def monte_carlo_all(logM, sigmaM, N_draw=10000,
                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  Mstar_min=0.1, 
                  Mstar_max=10):
    
    log_Ms = np.linspace(logM - 5*sigmaM, logM + 5*sigmaM, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    PHI_MBH = np.cumsum(log_norm(log_Ms, logM, sigmaM)*dMbh)/np.sum(log_norm(log_Ms, logM, sigmaM)*dMbh)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)


    psi_ = np.linspace(0.001, pi/2, N_psi)
    dpsi = psi_[1] - psi_[0]


    a_ = np.linspace(-0.9999, +0.9999, N_spin)
    da_ = a_[1] - a_[0]


    m_samples = np.full(N_draw, np.nan)
    a_samples = np.full(N_draw, np.nan)
    m_star_samples = np.full(N_draw, np.nan)
    psi_samples = np.full(N_draw, np.nan)

    print('Generating mass matrix......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)




    print('Generating samples......')
    n=0

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        q = np.random.uniform(0, 1, 1)
        j = np.argmin(abs(q - cos(psi_)))

        a_bh = np.random.uniform(0, 1, 1)
        i = np.argmin(abs(a_ - a_bh))

        p_ = max_mass_matrix[i, j] > (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5)

        if p_:
            m_samples[n] = l_m_bh 
            a_samples[n] = a_bh
            m_star_samples[n] = m_star
            psi_samples[n] = np.arccos(q)
            n+=1
            pbar.update()


    a_samples = a_samples[a_samples==a_samples]
    m_samples = m_samples[m_samples==m_samples]
    psi_samples = psi_samples[psi_samples==psi_samples]
    m_star_samples = m_star_samples[m_star_samples==m_star_samples]

    return a_samples, m_samples, m_star_samples, psi_samples
        

def tde_mass_dist(N_draw, 
                  black_hole_mass_dist=None, 
                  M_bh_min = 1e5, 
                  M_bh_max = 1e9, 
                  N_bh=1000, 
                  bh_spin = 0.9, 
                  N_star = 100, 
                  N_spin = 300, 
                  N_psi = 200, 
                  Mstar_min=0.1, 
                  Mstar_max=10):

    

    log_Ms = np.linspace(np.log10(M_bh_min), np.log10(M_bh_max), N_bh+1)
    d_Mbhs = log_Ms[1:] - log_Ms[:-1]

    p_MBHs = black_hole_mass_dist(10**log_Ms * Ms)
    log_Ms = log_Ms[:-1]


    PHI_MBH = np.cumsum(p_MBHs*d_Mbhs)/np.sum(p_MBHs*d_Mbhs)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)

    psi_ = np.linspace(0.001, pi/2, N_psi)
    dpsi = psi_[1] - psi_[0]


    a_ = np.array([bh_spin])

    print('Generating mass matrix......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)

    print('Generating samples......')
    n=0

    m_samples = np.full(N_draw, np.nan)

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        q = np.random.uniform(0, 1, 1)
        j = np.argmin(abs(q - cos(psi_)))

        # a_bh = np.random.uniform(0, 1, 1)
        i = 0#np.argmin(abs(a_ - a_bh))

        p_ = max_mass_matrix[i, j] > (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5)

        if p_:
            m_samples[n] = l_m_bh 
            # a_samples[n] = a_bh
            # m_star_samples[n] = m_star
            # psi_samples[n] = np.arccos(q)
            n+=1
            pbar.update()


    m_samples = m_samples[m_samples==m_samples]

    return m_samples


def one_d_mass_dist(prior_Mbh=None, prior_spin=0., 
                  M_bh_min = 1e5, 
                  M_bh_max = 1e9, 
                  N_bh=150, 
                  N_star = 100, 
                  N_psi = 200, 
                  N_spin = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=10):
    
    
    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]


    psi_ = np.linspace(0.001, pi/2, N_psi)
    dpsi = psi_[1] - psi_[0]


    a_ = np.linspace(-0.9999, +0.9999, N_spin)
    da_ = a_[1] - a_[0]

    if type(prior_spin) != type(log_norm):
        a_ = np.array([prior_spin])

    print('Generating mass matrix......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)


    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5)

    mass_mass_matrix = [[] for _ in range(len(a_))]

    for k in range(len(a_)):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix




    log_Ms = np.linspace(np.log10(M_bh_min), np.log10(M_bh_max), N_bh+1)
    d_Mbhs = log_Ms[1:] - log_Ms[:-1]

    p_MBHs = prior_Mbh(10**log_Ms * Ms)
    log_Ms = log_Ms[:-1]

    p_bh = np.zeros(len(log_Ms))

    print('Getting 1D mass distribution.....')
    for i, l_m_bh in enumerate(tqdm(log_Ms)):
        p = 0
        m_bh = 10**l_m_bh * Ms
        for j, a in enumerate(range(len(a_))):
            p_ = np.heaviside(mass_mass_matrix[j] - m_bh, 1)
            if len(a_) > 1:
                p += np.sum(np.sum(p_ * psd * dMs, axis=1) * sin(psi_) * dpsi, axis=0) * p_MBHs[i] * d_Mbhs[i] * prior_spin(a) * da_
            else:
                p += np.sum(np.sum(p_ * psd * dMs, axis=1) * sin(psi_) * dpsi, axis=0) * p_MBHs[i] * d_Mbhs[i]
    
        p_bh[i] = p

    p_bh = p_bh/sum(p_bh * d_Mbhs)
    
    return p_bh, log_Ms




if __name__ == "__main__":
    main()