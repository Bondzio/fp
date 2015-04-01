def fit_n_plot_ps(theta, fit_boundaries, p0):
    '''
    Plots coincident electrons from PVC scintillator
    - measured at theta (in degress)
    
    Needs global variables:
    rate_bg, rate_rnd, show_fig, save_fig
    Needs background and random coincidences to be defined
    globally as rate_bg and rate_rnd, respectively   
    Saves to pdf and png if specified.
    '''
    #### GET DATA  ####
    file_name = "coin_ps_" + str(theta)
    file_in = npy_dir + "coin_ps_" + str(theta) + '.npy'
    x, y = np.load(file_in)
    t = 3600
    rate = y / t
    #rate = rate - rate_bg
    #rate = rate - rate_rnd
    rate[rate < 0] = 0
    y = rate    # Continue to work with the rate!

    ###### PLOTTING #################
    if show_fig:
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("PVC scintillator, coincident; angle: %i$^\circ$"%theta)
        ax1.plot(x, y, '.', alpha=0.9, label='data')
        ax1.set_xlabel("Channel")
        ax1.set_ylabel("Rate / $s^{-1}$")
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(0,)
        ax1.legend(loc=1)
        ax1.grid(True)
        if save_fig:
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")

    return
