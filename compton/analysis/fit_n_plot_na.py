def fit_n_plot_na(theta, fit_boundaries, p0, fit=True):
    '''
    Fits coincident photons from NaI scintillator
    - measured at theta (in degress)
    - fit range: fit_boundaries = [x_min, x_max] 
    - initial guess: p0  = [A, x_peak, sigma, offset]
    
    Needs global variables:
    rate_bg, rate_rnd, show_fig, save_fig

    Uses gauss plus offset for fit.
    Needs background and random coincidences to be defined
    globally as rate_bg and rate_rnd, respectively   
    Further plots the result.
    Saves to pdf and png if specified.
    Returns coefficients as correlated variables c
    (use uc.covariance_matrix to obtain covariance matrix!).
    '''
    #### GET DATA  ####
    file_name = "coin_na_" + str(theta)
    file_in = npy_dir + "coin_na_" + str(theta) + '.npy'
    y = np.load(file_in)
    y_e = un.uarray(y, np.sqrt(y))
    t = 3600
    rate = y / t
    #rate = rate - rate_bg
    #rate = rate - rate_rnd
    rate[rate < 0] = 0
    y = rate    # Continue to work with the rate!

    rate_e = y_e / t # - rate_bg_e - rate_rnd_e
    rate_e[rate < 0] = 0 


    # Rebinning: 1/16 of number of bins
    z = y[:-14]
    z = z.reshape([len(z) / 16, 16])
    z = np.sum(z, axis=1)
    y = z

    z_e = rate_e[:-14]
    z_e = z_e.reshape([len(z_e) / 16, 16])
    z_e = np.sum(z_e, axis=1)
    y_e = z_e
    np.save(npy_dir + 'na_rate_' + str(theta), y_e)

    x = np.arange(len(y))
    y_filtered = sav(y, 201, 7)

    ###### FIT PEAK #################
    x_min, x_max = fit_boundaries
    mask = (x >x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = y[mask]
    y_sigma = np.sqrt(un.std_devs(y_e[mask]))
    if fit:
        coeff, cov = curve_fit(gauss, x_fit, y_fit, p0=p0)
#                sigma=y_sigma, absolute_sigma=True)
        fit_peak = gauss(x_fit, *coeff)
        #x_c = coeff[1]
        c = uc.correlated_values(coeff, cov)
    else:
        fit_peak = gauss(x_fit, *p0)      # visualize initial guess
        c = 0

    ###### PLOTTING #################
    if show_fig:
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("NaI scintillator, coincident; angle: %i$^\circ$"%theta)
        ax1.plot(x, y, '.', alpha=0.9, label='data')
        peak, = ax1.plot(x_fit, fit_peak, '-', alpha=0.8, label='peak fit')
        #ax1.plot([x_c] * 2, [0, max(y)*1.5], '--', c=peak.get_color(), label='peak')
        ax1.set_xlabel("Channel")
        ax1.set_ylabel("Rate / $s^{-1}$")
        ax1.set_xlim(0, 800)
        ax1.set_ylim(0,)
        ax1.legend(loc=1)
        ax1.grid(True)
        if save_fig:
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")

    return(c)
