import numpy as np
import pylab as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','sans-serif':['helvetica']})

def csv_to_npy(n):
    '''
    create numpy arrays for t, output signal "signal" and input signal "sine".
    '''
    input_dir = "../data/lock_in/"
    output_dir = input_dir + "npy/"
    filename = "LOCK" + str(n)
    t = []
    signal = []
    file = open(input_dir + filename + ".csv")   # read data of signal from csv table
    for i, line in enumerate(file):
        if not i==0:
            x = line.split(",")
            t.append(float(x[0]))
            signal.append(float(x[1]))
    file.close()
    t = np.array(t)
    signal = np.array(signal)
    return t, signal

def plot_lock_in(n, js, captions, fig_name, figsize):
    # Plot U(t) as n stacked plots
    fig, ax= plt.subplots(n, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)          # create overlap
    xticklabels = ax[0].get_xticklabels()    
    plt.setp(xticklabels, visible=False)    # hide the xticks of the upper plot
    labels = [
            '$T = 0.00$ for $T_\mathrm{step} = 2.5$ ms, Mode = Norm, $T_0$ = Norm',
            '$T = 5.00$ for $T_\mathrm{step} = 25$ ms, Mode = Norm, $T_0$ = Norm', 
            '$T = 5.00$ for $T_\mathrm{step} = 25$ ms, Mode = Inv, $T_0$ = Norm',
            '$T = 1.70$ for $T_\mathrm{step} = 25$ ms, Mode = Norm, $T_0$ = Norm', 
            'rectangular gate signal'
            ]
    # plotting measured signal
    for k, j in enumerate(js):
        t, sig = csv_to_npy(j)
        #signal_label = str(np.mean(sig))
        signal_label = labels[k]
        ax[k].plot(t, sig, '-', label=signal_label, linewidth=0.5)                # plot signal
        ax[k].legend(loc=1, frameon=True, fontsize=12)
        ax[k].grid(b=True)
        ax[k].set_xlim(min(t), max(t))
        ax[k].set_ylim(top=(ax[k].get_ylim()[1]*1.8)) # set ylim such that legend doesn't overlap signal
        ax[k].set_ylabel('$U(t) \, / \, \mathrm{V}$')
        ax[k].set_yticks(ax[k].get_yticks()[1:-1]) # hide the overlapping ytick of the upper plot
    ax[n-1].set_xlabel('$t \, / \, \mathrm{s}$')
    if save_fig:
        fig.savefig(fig_dir + fig_name + ".pdf")
        f1.write('\\begin{figure}\n')
        f1.write('\t\includegraphics[width=\\textwidth]{figures/%s.pdf}\n'%fig_name)
        f1.write('\t\caption{\n')
        for caption in captions:
            f1.write('\t\t' + caption + '\n')
        f1.write('\t\t}\n\t\label{fig:%s}\n'%fig_name)
        f1.write('\end{figure}\n\n')
    if show_fig:
        fig.show()
    return 0

#####################################################################################################
show_fig = 0
save_fig = 1
plt.close('all')
fig_dir = '../figures/'
# Specify parameters:
# js: 
#   (19, 20) absorption 
#   21 - 27 garbage
#   (28, 29), (30, 31), (32, 33), (34, 35, 36) = (sawtooth, no integration (, integration))
#   (37, 38), ..., (47, 48) = (no integration, integration)
#   49, 50 modulation,
#   (51, 52) =  multiplication sawtooth sine (no intergration, integration)
#   (53, 54)  = (sawtooth, measured signal)
js = [29, 31, 33, 35, 28]
n = len(js)
I = 2.96
nu = 16.8220
captions = [
        r"Lock-in phase calibration.",
        r"$T$ indicates the value shown on the potentiometer 'Time Const', ",
        r"the following three parameters correspond to the toggle switches located on the lock-in analyser.",
        r"One can make the following observations: For $T = 0$, the phase difference $\Delta \phi$ is not zero;",
        r"the inversion does not change the signal's form but only the sign; The phase differences corresponding ",
        r"to the lower three signals are $\Delta \phi = 2\pi, \pi, \pi / 2$, respectively.",
        r"The error on $T$ is given by $s_T = 0.03$."
    ]
fig_name = "phase"
figsize = (8.09,2*5)

if save_fig: f1 = open(fig_name + '.tex', 'w+')
plot_lock_in(n, js, captions, fig_name, figsize)
if save_fig: f1.close()
