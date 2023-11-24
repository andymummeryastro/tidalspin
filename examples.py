import numpy as np 
from astropy import constants 
import matplotlib.pyplot as plt
import matplotlib
cmap = matplotlib.cm.get_cmap('coolwarm')

import tidalspin as tp 

Ms = constants.M_sun.value
Rs = constants.R_sun.value
c = constants.c.value
G = constants.G.value


def main():

    tp.format_plots_nicely()

    fig = example_one()

    # fig, fig2 = example_two()

    # fig = example_three()

    # fig, fig2 = example_four()

    # fig = example_five()

    # fig = example_six()

    # fig, fig2 = example_seven()

    plt.show()


def example_one():
    logM = 8.5## prior estimate for peak of log_10(black hole mass)
    sigmaM = 0.3## uncertainty in log_10(black hole mass)

    prior_MBH = lambda lm: tp.log_norm(lm, logM, sigmaM) * (10**lm/1e8)**0.03 * np.exp(-(10**lm/(6.4e7))**0.49)## Assume log-normal mass prior. 


    hm, mm = tp.get_hills_masses()## Get Hills masses for solar mass star (for comparison)

    p_a, a = tp.one_d_spin_dist(prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10, max_mass_matrix=mm)## Get 1D spin posterior 
    p_m, m = tp.one_d_mass_dist(prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10, max_mass_matrix=mm)## Get 1D mass posterior
    a_samples, m_samples, _, _ = tp.monte_carlo_all(prior_Mbh=prior_MBH,  log_Mbh_min=7, log_Mbh_max=10, N_draw=100000, max_mass_matrix=mm)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax2.axis('off')

    ax3.scatter(m_samples, abs(a_samples), marker='.', color='blue', alpha=0.02, rasterized=True)
    ax3.set_xlabel(r'$\log_{10}M_\bullet/M_\odot$', fontsize=30)
    ax3.set_ylabel(r'$|a_\bullet|$', fontsize=30)
    ax3.plot(np.log10(hm[150:]/Ms), a[150:], ls='-.', c='k')
    ax3.set_ylim(0, 1)
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

    ax4.hist(abs(a_samples), bins=30, density=True, color='blue')
    ax4.plot(a[150:], p_a[150:] + p_a[150 - 1 :: -1], ls='--', c='r')

    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r'$p(|a_\bullet|)$', fontsize=30, rotation=270, labelpad=40)
    ax4.set_xlabel(r'$|a_\bullet|$', fontsize=30)
    ax4.set_xlim(0, 1)
    ax4.yaxis.tick_right()
    ax4.set_yticks([])
    ax4.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    return fig 


def example_two():

    def SMBHMassDistributionFunction(log_mBHs, Mc=1e6*Ms, alpha=0):
        mBHs = 10**log_mBHs * Ms
        m = mBHs/(1e6*Ms)
        ps = m**(alpha)/(1 + (Mc/mBHs)**(2-alpha)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
        return ps


    hm, mm = tp.get_hills_masses()## Get Hills masses for solar mass star (for comparison)

    N_bh = 300
    N_draw = 10000

    fig = plt.figure()
    ax = fig.add_subplot()
    
    prior_spins = lambda a: np.exp(-(a-0)**2.0/0.01**2.0)
    _, m_tde, _, _ = tp.monte_carlo_all(N_draw=N_draw, prior_Mbh=SMBHMassDistributionFunction, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm) 
    ptde, _ = np.histogram(m_tde, bins=50, density=True)
    ax.hist(m_tde, bins=100, density=True, histtype=u'step', color='blue')

    X_REAL1 = np.sort(m_tde)
    F_REAL1 = np.array(range(len(m_tde)))/float(len(m_tde))

    
    prior_spins = lambda a: np.exp(-(a-0.5)**2.0/0.01**2.0)
    _, m_tde, _, _ = tp.monte_carlo_all(N_draw=N_draw, prior_Mbh=SMBHMassDistributionFunction, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm) 
    ptde, _ = np.histogram(m_tde, bins=50, density=True)
    ax.hist(m_tde, bins=100, density=True, histtype=u'step', color='red')

    X_REAL2 = np.sort(m_tde)
    F_REAL2 = np.array(range(len(m_tde)))/float(len(m_tde))

    prior_spins = lambda a: np.exp(-(a-0.75)**2.0/0.01**2.0)
    _, m_tde, _, _ = tp.monte_carlo_all(N_draw=N_draw, prior_Mbh=SMBHMassDistributionFunction, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm) 
    ptde, _ = np.histogram(m_tde, bins=50, density=True)
    ax.hist(m_tde, bins=100, density=True, histtype=u'step', color='green')

    X_REAL3 = np.sort(m_tde)
    F_REAL3 = np.array(range(len(m_tde)))/float(len(m_tde))

    prior_spins = lambda a: np.exp(-(a-0.9)**2.0/0.01**2.0)
    _, m_tde, _, _ = tp.monte_carlo_all(N_draw=N_draw, prior_Mbh=SMBHMassDistributionFunction, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm)  
    ptde, _ = np.histogram(m_tde, bins=50, density=True)
    ax.hist(m_tde, bins=100, density=True, histtype=u'step', color='purple')

    X_REAL4 = np.sort(m_tde)
    F_REAL4 = np.array(range(len(m_tde)))/float(len(m_tde))

    prior_spins = lambda a: np.exp(-(a-0.999)**2.0/0.01**2.0)
    _, m_tde, _, _ = tp.monte_carlo_all(N_draw=N_draw, prior_Mbh=SMBHMassDistributionFunction, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm)  
    ptde, _ = np.histogram(m_tde, bins=50, density=True)
    ax.hist(m_tde, bins=100, density=True, histtype=u'step', color='black')

    X_REAL5 = np.sort(m_tde)
    F_REAL5 = np.array(range(len(m_tde)))/float(len(m_tde))

    ax.set_yscale('log')


    prior_spins = lambda a: np.exp(-(a-0)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    ax.semilogy(m_tde, p_tde, c='blue', ls='--')
    
    prior_spins = lambda a: np.exp(-(a-0.5)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    ax.semilogy(m_tde, p_tde, c='red', ls='--')
    
    prior_spins = lambda a: np.exp(-(a-0.75)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    ax.semilogy(m_tde, p_tde, c='green', ls='--')

    prior_spins = lambda a: np.exp(-(a-0.9)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    ax.semilogy(m_tde, p_tde, c='purple', ls='--')

    prior_spins = lambda a: np.exp(-(a-0.999)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    ax.semilogy(m_tde, p_tde, c='black', ls='--')

    ax.set_ylim(1e-5, 10)

    ax.set_xlabel(r'$\log_{10} M_\bullet/M_\odot$')
    ax.set_ylabel(r'$p(\log_{10} M_\bullet/M_\odot$)')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    for (X,F) in zip([X_REAL1, X_REAL2, X_REAL3, X_REAL4, X_REAL5], [F_REAL1, F_REAL2, F_REAL3, F_REAL4, F_REAL5]):
        ax2.plot(X, F)

    ax2.set_xlabel(r'$\log_{10} M_\bullet/M_\odot$')
    ax2.set_ylabel(r'$\Phi(\log_{10} M_\bullet/M_\odot$)')
    return fig, fig2


def example_three():
    def SMBHMassDistributionFunction(logmBHs, Mc=1e3*Ms, alpha=0.03):
        mBHs= 10**logmBHs * Ms
        m = mBHs/(1e6*Ms)
        ps = m**(alpha)/(1 + (Mc/mBHs)**(2-alpha)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
        return ps

    N_bh = 300
    mbhs = np.linspace(5, 10, N_bh) 
    p_no_tde = SMBHMassDistributionFunction(mbhs)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    hm, mm = tp.get_hills_masses()


    prior_spins = lambda a: np.exp(-(abs(a)-0)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, N_bh=N_bh, max_mass_matrix=mm) ## Get 1D mass posterior
    ax1.semilogy(m_tde, p_tde, label=r'$| \bar a_\bullet | = 0$', c=cmap(1/5))

    
    prior_spins = lambda a: np.exp(-(abs(a)-0.5)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, N_bh=N_bh, max_mass_matrix=mm) ## Get 1D mass posterior
    ax1.semilogy(m_tde, p_tde, label=r'$| \bar a_\bullet | = 0.5$', c=cmap(2/5))

    prior_spins = lambda a: np.exp(-(abs(a)-0.75)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, N_bh=N_bh, max_mass_matrix=mm) ## Get 1D mass posterior
    ax1.semilogy(m_tde, p_tde, label=r'$| \bar a_\bullet | = 0.75$', c=cmap(3/5))
    

    prior_spins = lambda a: np.exp(-(abs(a)-0.9)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, N_bh=N_bh, max_mass_matrix=mm) ## Get 1D mass posterior
    ax1.semilogy(m_tde, p_tde, label=r'$| \bar a_\bullet | = 0.9$', c=cmap(4/5))


    prior_spins = lambda a: np.exp(-(abs(a)-0.999)**2.0/0.01**2.0)
    p_tde, m_tde = tp.one_d_mass_dist(prior_Mbh=SMBHMassDistributionFunction, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, N_bh=N_bh, max_mass_matrix=mm) ## Get 1D mass posterior
    ax1.semilogy(m_tde, p_tde, label=r'$| \bar a_\bullet | = 0.999$', c=cmap(5/5))


    ax1.semilogy(mbhs, p_no_tde / np.sum(p_no_tde * (mbhs[1]-mbhs[0])), '--', c='k', label=r'Intrinisic')

    ax1.set_ylim(1e-5, 10)

    ax1.set_xlabel(r'$\log_{10}M_\bullet/M_\odot$')
    ax1.set_ylabel(r'$N_{\rm TDE}^{-1} \times {\rm d}N_{\rm TDE}/{\rm d}\log_{10}\left(M_\bullet/M_\odot\right)$')
    

    ax1.legend()

    return fig1


def example_four():

    def SMBHMassDistributionFunction(logmBHs, Mc=5e5*Ms, slope=-0.5):
        mBHs= 10**logmBHs * Ms
        m = mBHs/(1e6*Ms)
        ps = m**(slope)/(1 + (Mc/mBHs)**(4-slope)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
        return ps



    beta = 1/0.98#0.567
    alpha = -6.52/0.98 + 43#39.45
    epsilon = 0.05#0.36

    lum_dist = lambda l, m: np.exp(-(l - (alpha + beta * m))**2.0/(2*epsilon**2.0)) * 1/(1+(10**42.3/10**l)**4)#np.heaviside(l-42, 0)
    lums = np.linspace(40, 47, 400)

    hm, mm = tp.get_hills_masses()

    Mc = 1e3#3.8e5 * Ms 
    slope = beta * 1.5 - 0.4 - 1.1

    prior_MBH = lambda logM: SMBHMassDistributionFunction(logM, Mc=Mc, slope=slope)

    fig1, fig2 = plt.figure(), plt.figure()
    ax1, ax2 = fig1.add_subplot(), fig2.add_subplot()
    
    prior_spins = lambda a: np.exp(-(abs(a)-0)**2.0/0.01**2.0)
    p_lums = tp.one_d_observable_dist(max_mass_matrix=mm, observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=4, log_Mbh_max=10, prior_spin=prior_spins)## Get 1D mass posterior
    ax1.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$| \bar a_\bullet | = 0$', c='orange')

    cdf_tde = np.cumsum(p_lums * (lums[1] - lums[0]))/(np.sum(p_lums) * (lums[1] - lums[0]))
    ax2.plot(lums, cdf_tde, label=r'$| \bar a_\bullet | = 0$', c='orange', ls='--')


    prior_spins = lambda a: np.exp(-(abs(a)-0.8)**2.0/0.01**2.0)
    p_lums = tp.one_d_observable_dist(max_mass_matrix=mm, observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=4, log_Mbh_max=10, prior_spin=prior_spins)## Get 1D mass posterior
    ax1.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$| \bar a_\bullet | = 0.8$', c='purple')

    
    cdf_tde = np.cumsum(p_lums * (lums[1] - lums[0]))/(np.sum(p_lums) * (lums[1] - lums[0]))
    ax2.plot(lums, cdf_tde, label=r'$| \bar a_\bullet | = 0.8$', c='purple', ls='--')

    prior_spins = lambda a: np.exp(-(abs(a)-0.95)**2.0/0.01**2.0)
    p_lums = tp.one_d_observable_dist(max_mass_matrix=mm, observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=4, log_Mbh_max=10, prior_spin=prior_spins)## Get 1D mass posterior
    ax1.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$| \bar a_\bullet | = 0.9$', color='seagreen')

    cdf_tde = np.cumsum(p_lums * (lums[1] - lums[0]))/(np.sum(p_lums) * (lums[1] - lums[0]))
    ax2.plot(lums, cdf_tde, label=r'$| \bar a_\bullet | = 0.9$', c='seagreen', ls='--')

    prior_spins = lambda a: np.exp(-(abs(a)-0.999)**2.0/0.01**2.0)
    p_lums = tp.one_d_observable_dist(max_mass_matrix=mm, observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=4, log_Mbh_max=10, prior_spin=prior_spins)## Get 1D mass posterior
    ax1.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$| \bar a_\bullet | = 0.999$', c='red')

    cdf_tde = np.cumsum(p_lums * (lums[1] - lums[0]))/(np.sum(p_lums) * (lums[1] - lums[0]))
    ax2.plot(lums, cdf_tde, label=r'$| \bar a_\bullet | = 0.999$', c='red', ls='--')

    ax1.set_xlabel(r'$\log_{10} L_{g, {\rm peak}}$ [erg  s$^{-1}$]')
    ax1.set_ylabel(r'$N_{\rm TDE}^{-1} \times {\rm d} N_{\rm TDE} / {\rm d}\log_{10} L_{g, {\rm peak}}$')        

    ax1.set_ylim(1e-5, 10)
    ax1.set_xlim(41.5, 45.5)
    
    ax1.legend(ncol=3)

    ax2.set_xlabel(r'$\log_{10} L_{g, {\rm peak}}$ [erg  s$^{-1}$]')
    ax2.set_ylabel(r'$\Phi\left(\log_{10} L_{g, {\rm peak}}\right)$')  
    ax2.set_xlim(41.5, 45.5)
    ax2.grid()

    return fig1, fig2


def example_five():
    def SMBHMassDistributionFunction(logmBHs, Mc=5e5*Ms, slope=-0.5):
        mBHs= 10**logmBHs * Ms
        m = mBHs/(1e6*Ms)
        ps = m**(slope)/(1 + (Mc/mBHs)**(2-slope)) * np.exp(- np.power(mBHs/(6.4e7*Ms), +0.49))##Shankar 04
        return ps


    N_bh = 300
    N_draw = 10000

    beta = 1
    alpha = 36
    epsilon = 0.3

    lum_func = lambda m: (alpha + beta * m) + np.random.normal(0, epsilon)

    lum_dist = lambda l, m: np.exp(-(l - (alpha + beta * m))**2.0/(2*epsilon**2.0))
    
    lums = np.linspace(40, 47, 400)

    hm, mm = tp.get_hills_masses()

    Mc = 1e6 * Ms 
    slope = -0.5

    prior_MBH = lambda logM: SMBHMassDistributionFunction(logM, Mc=Mc, slope=slope)
    
    prior_spins = lambda a: np.exp(-(a-0)**2.0/0.01**2.0)

    fig = plt.figure()
    ax = fig.add_subplot()
    
    obs_tde, _, m_tde, _, _ = tp.monte_carlo_observable(observe_func=lum_func, N_draw=N_draw, prior_Mbh=prior_MBH, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm) 
    plt.hist(obs_tde, bins=100, density=True, histtype=u'step', color='blue')
    p_lums = tp.one_d_observable_dist(observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    plt.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$a_\bullet = 0$', c='blue', ls='--')
    
    prior_spins = lambda a: np.exp(-(a-0.998)**2.0/0.01**2.0)
    obs_tde, _, m_tde, _, _ = tp.monte_carlo_observable(observe_func=lum_func, N_draw=N_draw, prior_Mbh=prior_MBH, log_Mbh_max=10, log_Mbh_min=5, N_bh=N_bh, prior_spin=prior_spins, max_mass_matrix=mm) 
    plt.hist(obs_tde, bins=100, density=True, histtype=u'step', color='red')
    p_lums = tp.one_d_observable_dist(observe_dist=lum_dist, observe_values=lums, prior_Mbh=prior_MBH, log_Mbh_min=5, log_Mbh_max=10, prior_spin=prior_spins, max_mass_matrix=mm) ## Get 1D mass posterior
    plt.semilogy(lums, p_lums/(np.sum(p_lums) * (lums[1] - lums[0])), label=r'$a_\bullet = 0.998$', c='red', ls='--')


    plt.ylim(1e-5, 10)
    plt.xlim(40.5, 46.5)
    plt.legend()
    plt.xlabel(r'$\log_{10} L_{g, {\rm peak}}$ [erg  s$^{-1}$]')
    plt.ylabel(r'${\rm d} N_{\rm TDE} / {\rm d}\log_{10} L_{g, {\rm peak}}$ [erg$^{-1}$  s]')

    return fig


def example_six():
    fig = plt.figure()
    ax = fig.add_subplot()
    
    hm, mm = tp.get_hills_masses(N_spin=500)
    
    for nn, logM in enumerate([7.6, 7.9, 8.2, 8.5, 8.8]):
        sigmaM = 0.3## uncertainty in log_10(black hole mass)

        prior_MBH = lambda lm: tp.log_norm(lm, logM, sigmaM) * (10**lm/1e8)**0.03 * np.exp(-(10**lm/(6.4e7))**0.49)## Assume log-normal mass prior. 

        prior_spins = lambda a: np.ones_like(a)## agnostic spin prior. 

        p_a, a = tp.one_d_spin_dist(prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 

        ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='-', c=cmap(nn/4), label=r'$\log_{10}\mu_{M_\bullet}/M_\odot$ = %s'%logM)

    ax.legend()
    ax.grid()
    ax.set_xlabel(r'$|a_\bullet|$')
    ax.set_ylabel(r'$p(|a_\bullet|)$')

    return fig


def example_seven():

    fig = plt.figure()
    ax = fig.add_subplot()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    
    sigmaM = 0.3## uncertainty in log_10(black hole mass)
    logM = np.log10(tp.MH_sig(225))## ASASSN-15lh

    hm, mm = tp.get_hills_masses(N_spin=500)

    prior_MBH = lambda lm: tp.log_norm(lm, logM, sigmaM) * (10**lm/1e8)**0.03 * np.exp(-(10**lm/(6.4e7))**0.49)## Assume log-normal mass prior. 

    prior_spins = lambda a: np.ones_like(a)## agnostic spin prior. 

    likelihood = lambda x: np.ones_like(x)

    xs = np.logspace(-3, 0, 100)

    p_a, a = tp.one_d_spin_dist(likelihood=likelihood, prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, N_spin=500, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 
    ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='--', c='k', label=r'$f(x) = 1$')
    ax2.semilogx(xs, likelihood(xs), ls='--', c='k', label=r'$f(x) = 1$')

    likelihood = lambda x: 1 - x**2
    p_a, a = tp.one_d_spin_dist(likelihood=likelihood, prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, N_spin=500, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 
    ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='-', c=cmap(1/4), label=r'$f(x) = 1 - x^2$')
    ax2.semilogx(xs, likelihood(xs), ls='-', c=cmap(1/4), label=r'$f(x) = 1-x^2$')

    likelihood = lambda x: 1 - np.exp(-1/x)
    p_a, a = tp.one_d_spin_dist(likelihood=likelihood, prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, N_spin=500, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 
    ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='-', c=cmap(2/4), label=r'$f(x) = 1 - \exp(-1/x)$')
    ax2.semilogx(xs, likelihood(xs), ls='-', c=cmap(2/4), label=r'$f(x) = 1-\exp(-1/x)$')

    likelihood = lambda x: np.exp(-x**2)
    p_a, a = tp.one_d_spin_dist(likelihood=likelihood, prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, N_spin=500, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 
    ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='-', c=cmap(3/4), label=r'$f(x) = \exp(-x^2)$')
    ax2.semilogx(xs, likelihood(xs), ls='-', c=cmap(3/4), label=r'$f(x) = \exp(-x^2)$')

    likelihood = lambda x: 1 - x
    p_a, a = tp.one_d_spin_dist(likelihood=likelihood, prior_Mbh=prior_MBH, prior_spin=prior_spins, log_Mbh_min=5, log_Mbh_max=10, N_spin=500, Mstar_max=1, max_mass_matrix=mm)## Get 1D spin posterior 
    ax.plot(a[250:], p_a[249::-1]+p_a[250:], ls='-', c=cmap(4/4), label=r'$f(x) = 1 - x$')
    ax2.semilogx(xs, likelihood(xs), ls='-', c=cmap(4/4), label=r'$f(x) = 1- x$')

    ax.legend()
    ax.grid()
    ax.set_xlabel(r'$|a_\bullet|$')
    ax.set_ylabel(r'$p(|a_\bullet|)$')

    ax2.legend()
    ax2.grid()
    ax2.set_xlabel(r'$M_\bullet/\widetilde M_\bullet$')
    ax2.set_ylabel(r'$f$')

    return fig, fig2


if __name__ == "__main__":
    main()