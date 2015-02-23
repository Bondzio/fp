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
    fig, ax= plt.subplots(n, 1, sharex=False, figsize=figsize)
    fig.subplots_adjust(hspace=0.4)          # create overlap
    xticklabels = ax[0].get_xticklabels()    
    plt.setp(xticklabels, visible=True)    # hide the xticks of the upper plot
    labels = [
            "sine modulation (zoom in of the graph below)",
            "sine modulated saw tooth signal",
            "output signal (noise)"
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
        if k == 0:
            ax[k].set_xlim(0, 2)
        ylim = [(-1.5, 0), (-1.5, 2), (-0.04, 0.04)][k]
        ax[k].set_ylim(ylim)
        #ax[k].set_ylim(top=(ax[k].get_ylim()[1])) # set ylim such that legend doesn't overlap signal
        #ax[k].set_yticks(ax[k].get_yticks()[1:-1]) # hide the overlapping ytick of the upper plot
        ax[k].set_ylabel('$U(t) \, / \, \mathrm{V}$')
        ax[k].set_xlabel('$t \, / \, \mathrm{s}$')
    if save_fig:
        fig.savefig(fig_dir + fig_name + ".pdf")
        f1.write('\\begin{figure}\n')
        f1.write('\t\includegraphics[width=\\textwidth]{figures/%s.pdf}\n'%fig_name)
        f1.write('\t\caption{\n')
        for caption in captions:
            f1.write('\t\t' + caption + '\n')
        f1.write('\t\t}\n\t\label{fig:lock_in_%s}\n'%fig_name)
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
#   (53, 54)  = (sawtooth, measured signal)
js = [53, 53, 54]
n = len(js)
captions = [
        r"Example for failure of the lock-in method.",
        r"The upper two graphs show the sine modulated saw tooth signal", 
        r"in two different ranges for $t$. The zoomed range of the uppermost ",
        r"graph can be used to calculate the frequency of the sine modulation.", 
        r"The output signal is basically noise, no absorption signal can be observed."
    ]
fig_name = "example"
figsize = (8.09,2*5)

if save_fig: f1 = open(fig_name+ '.tex', 'w+')
plot_lock_in(n, js, captions, fig_name, figsize)
if save_fig: f1.close()
