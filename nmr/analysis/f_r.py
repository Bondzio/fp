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
    input_dir = "../data/fr/"
    output_dir = input_dir + "npy/"
    filename = "fr" + str(n)
    t = []
    signal = []
    sine = []
    file = open(input_dir + filename + "a" + ".csv")   # read data of signal from csv table
    for i, line in enumerate(file):
        if not i==0:
            x = line.split(",")
            t.append(float(x[0]))
            signal.append(float(x[1]))
    file.close()
    file = open(input_dir + filename + "b" + ".csv")   # read data of sine from csv table
    for i, line in enumerate(file):
        if not i==0:
            x = line.split(",")
            sine.append(float(x[1]))
    file.close()
    t = np.array(t)
    signal = np.array(signal)
    sine = np.array(sine)
    np.save(output_dir + filename + "_t", t)
    np.save(output_dir + filename + "_signal", signal)
    np.save(output_dir + filename + "_sine", sine)
    return t, signal, sine

def plot_signal(n, js, sample, I, nu, captions, fig_name, figsize):
    # Plot U(t) as n stacked plots
    fig, ax= plt.subplots(n+1, 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)          # create overlap
    xticklabels = ax[0].get_xticklabels()    
    plt.setp(xticklabels, visible=False)    # hide the xticks of the upper plot
    # plotting measured signal
    for k, j in enumerate(js):
        t, sig, sine = csv_to_npy(j+1)
        signal_label = 'signal for %s with $I = %.2f$ A, $\\nu = %.4f$ MHz'%(sample[k], I[k], nu[k])
        ax[k].plot(t, sig, '-', label=signal_label, linewidth=0.5)                # plot signal
        ax[k].set_ylim(top=(ax[k].get_ylim()[1]*1.6)) # hide the overlapping ytick of the upper plot
    # plotting sine modulation
    ax[n].plot(t, sine, '-', label='sine modulation', linewidth=0.5)                # plot B-modulation
    ax[n].set_xlabel('$t \, / \, \mathrm{s}$')
    for k in range(n + 1):
        ax[k].legend(loc=1, frameon=True, fontsize=12)
        ax[k].grid(b=True)
        ax[k].set_xlim(min(t), max(t))
        ax[k].set_ylabel('$U(t) \, / \, \mathrm{V}$')
        ax[k].set_yticks(ax[k].get_yticks()[1:-1]) # hide the overlapping ytick of the upper plot
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
n = [4, 2, 2]
js = [range(4), [4, 5], [6, 7]]
F = "$^{19}$F, fluid"
T = "$^{19}$F, Teflon"
H =  "$^1$H"
G =  "glycol"
sample = [[F, F, T, F], [H, H], [G, G]]
I = [[3.23, 3.21, 3.21, 3.23], [2.94, 2.94], [2.94, 2.94]]
nu = [[16.8573, 16.8811, 16.8130, 16.8905], [16.8503, 16.8503], [16.8704, 16.8704]]
captions = [[
        'Absorption peaks of $^{19}$F, measured ',
        'at four different times. One can observe absorption peaks in each of',
        'the measured signals. However, due to the described problems with fine tuning,',
        'we were not able to produce equidistant peaks with a distance of half the,',
        'wavelength of the input signal, shown in the lowest graph.'
    ],[
        'Absorption peaks of $^1$H',
        'at two times, both with $B_\mathrm{measured} = 395$ mT.',
        'No equidistant absorption peaks at the zero intersect of the sine are seen.'
    ],[
        'Measured absorption peaks of glycol',
        'at two times, both with $B_\mathrm{measured} = 394$ mT.',
        'Again, no equidistant absorption peaks at the zero intersect of the sine are seen.'
    ]]
fig_name = ["f_r_F", "f_r_H", "f_r_glycol"]
figsize = [(8.09,2 * 5), (8.09,1.5*5), (8.09,1.5*5)]

if save_fig: f1 = open('plots_f_r.tex', 'w+')
for i in range(3):
    plot_signal(n[i], js[i], sample[i], I[i], nu[i], captions[i], fig_name[i], figsize[i])
if save_fig: f1.close()
