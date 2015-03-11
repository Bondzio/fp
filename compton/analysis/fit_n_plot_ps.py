def fit_n_plot_ps(theta, fit_boundaries, p0):
    '''
    Fits coincident electrons from PVC scintillator
    - measured at theta (in degress)
    - for a time of t = 3600 s
    - fit range: fit_boundaries = [x_min, x_max] 
    - initial guess: p0  = [A, x_step, sigma, offset]
    
    Needs global variables:
    rate_bg, rate_rnd, show_fig, save_fig

    Uses gauss plus offset for fit.
    Needs background and random coincidences to be defined
    globally as rate_bg and rate_rnd, respectively   
    Further plots the result.
    Saves to pdf and png if specified.
    Returns coefficients as correlated variables c
    (use uc.correlation_matrix to obtain covariance matrix!).
    '''
    #### GET DATA  ####
    file_name = "coin_ps_" + str(theta)
    file_in = npy_dir + "coin_ps_" + str(theta) + '.npy'
    x, y = np.load(file_in)
    rate = y / t
    #rate = rate - rate_bg
    rate = rate - rate_rnd
    rate[rate < 0] = 0
    y = rate    # Continue to work with the rate!

    ###### FIT PEAK #################
    x_min, x_max = fit_boundaries
    x_fit = x[(x >x_min) * (x < x_max)]
    y_fit = y[(x > x_min) * (x < x_max)]
    #fit_peak = gauss(x_fit, *p0)      # visualize initial guess
    coeff, cov = curve_fit(gauss, x_fit, y_fit, p0=p0)
    fit_peak = gauss(x_fit, *coeff)
    #x_c = coeff[1]

    ###### PLOTTING #################
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("PVC scintillator, coincident; angle: %i$^\circ$"%theta)
    ax1.plot(x, y, '.', alpha=0.9, label='data')
    peak, = ax1.plot(x_fit, fit_peak, '-', alpha=0.8, label='peak fit')
    #ax1.plot([x_c] * 2, [0, max(y)*1.5], '--', c=peak.get_color(), label='peak')
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Rate / $s^{-1}$")
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0,)
    ax1.legend(loc=1)
    ax1.grid(True)
    if show_fig:
        fig1.show()
    if save_fig:
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")

    c = uc.correlated_values(coeff, cov)
    return(c)
