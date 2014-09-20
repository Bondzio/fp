import numpy as np
import pylab as plt
import uncertainties as uc
from uncertainties import unumpy as un

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)

#############################################################################################
# general parameters
plot_show = 0
save_fig = 1
print_tab =1
print_tab_diff = 0
plt.close('all')
colors = ['b', 'g', 'r', 'pink']
labels = ['$v\'\' = %i \\rightarrow v\'$' %i for i in range(3)]
fig_dir = 'figures/'
golden_ratio = (1 + np.sqrt(5)) / 2
l1 = 7.0
figsize = [golden_ratio * l1, l1]

lambda_error = 0.3                        # uncertainty of spectrometer in lambda (nonimal = 0.3)

# since there are three progressions, we define all used quantities as dictionaries 
# which are then index with the number of progression ( = v'')
prog    = {}        # indices of elements of progression (with respect to array of all minima, "mins")
lm, cmm = [[],[],[]], [[],[],[]]    # wavelength in nm / wavenumber in cm^-1  of points of a progression
dl, dcm = [[],[],[]], [[],[],[]]    # differences of wavelength in nm / wavenumber in cm^-1 between to successive points of a progression 
# intensity and waves for iodine
# the lab data is saved for increasing wavelength lambda. As we are working in cm^-1, we need to keep in mind this reversed order
input_dir = 'data_npy/'
f_iod_i = input_dir + 'iodine_03_I.npy'
f_iod_l = input_dir + 'iodine_03_l.npy'
iod_i = np.load(f_iod_i)                  # intensity
iod_l = un.uarray(np.load(f_iod_l), lambda_error)    # wavelenght
l_min = 500     # minimum wavelenght at which to search for minima
l_max = 620     # maximum wavelenght at which to search for minima
l_range = (l_min < iod_l) & (iod_l < l_max)
iod_l = iod_l[l_range]
iod_i = iod_i[l_range]

# correction for air, linearly interpolated with  
# (n_l - 1) * 10 **(4) = [2.79, 2.78, 2.77, 2.76]  at corrisponding wavelenght [500, 540, 600, 680] nm   
l_lower = np.array([500, 540, 600])         # lower bounds for linear interpolation
l_upper = np.array([540, 600, 680])         # upper bounds for linear interpolation
for i in range(len(iod_l)):
    l = iod_l[i]                 
    j = np.where((l_lower < l.n) * (l.n <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    iod_l[i] = n_l * l
iod_cm = 1 / iod_l * 10 ** 7

# searching for local minima
b_max = 4   # maximum number of neighbours 
mins = np.r_[True, iod_i[1:] < iod_i[:-1]] & np.r_[iod_i[:-1] < iod_i[1:], [True]]
for b in range(2, b_max + 1):
        mins *= np.r_[[False]*b, iod_i[b:] < iod_i[:-b]] * \
                np.r_[iod_i[:-b] < iod_i[b:], b * [False]]
mins[[119, 479, 702, 754, 789, 825, 863]] = True        # manually adding band heads
lm_all = iod_l[mins]
cmm_all = iod_cm[mins]
dl_all = lm_all[1:] - lm_all[:-1]

# progression v'' = 0 -> v'
prog01 = np.where((cmm_all > 18300) * (cmm_all < 197000))[0]           # selectred point before intersect
prog02 = np.array([24, 26, 28, 30, 32, 34, 36, 38])      # selectred point in intersect
prog[0] = np.r_[prog01,  prog02]      # indices of progression 0, reffering to minima
# progression v'' = 1 -> v'
prog[1] = np.array([23, 25, 27, 29, 31, 33, 35, 37, 39, 40, 42, 44, 46])      # selectred point in intersect
# progression v'' = 2 -> v'
prog21 = np.array([41, 43, 45])      # until the last minima found.
prog22 = np.arange(47, 56)      # until the last minima found.
prog[2] = np.r_[prog21,  prog22]      # indices of progression 0, reffering to minima
for i in range(3):      # assigning values of wavenumbers and wavelength at minima
    lm[i] = lm_all[prog[i]] 
    cmm[i] = cmm_all[prog[i]]

# Plotting the absorbtion spectrum
fig0 = plt.figure(figsize=figsize)
#fig.suptitle('Iodine 2 molecule - absorbtion spectrum')
ax = plt.subplot(111)
title_spectrum = "Absorption spectrum"
#ax.set_title(title_spectrum)
ax.plot(un.nominal_values(iod_cm),iod_i, '-', color='chocolate')
for i in range(3):
    ax.plot(unv(cmm[i]),iod_i[mins][prog[i]], '.', label=labels[i])
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.grid(True)
ax.legend(loc='upper right')

# produce plots of critical regions, where points are chose by hand
fig1 = plt.figure(figsize=figsize)
#fig0.suptitle('Iodine 2 molecule - absorbtion spectrum at overlaps')
ax = plt.subplot(111)
title = "Overlap of progressions $v'=1 and v'=2$"
#ax.set_title(title)
ax.plot(un.nominal_values(iod_cm),iod_i, '-', color='chocolate')
for i in range(3):
    ax.plot(unv(cmm[i]),iod_i[mins][prog[i]], '.', label=labels[i])
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.set_xlim(17050, 17650)
ax.set_ylim(41500, 46000)
ax.legend(loc='upper left')

fig2 = plt.figure(figsize=figsize)
ax = plt.subplot(111)
title = "Overlap of progressions $v'=0 and v'=1$"
#ax.set_title(title)
ax.plot(un.nominal_values(iod_cm),iod_i, '-', color='chocolate')
for i in range(3):
    ax.plot(unv(cmm[i]),iod_i[mins][prog[i]], '.', label=labels[i])
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.set_xlim(17900, 18450)
ax.set_ylim(36700, 43700)
ax.legend(loc='lower left')

fig3 = plt.figure(figsize=figsize)
ax = plt.subplot(111)
title = "Overlap of progressions $v'=0 and v'=1$"
#ax.set_title(title)
ax.plot(un.nominal_values(iod_cm),iod_i, '-', color='chocolate')
for i in range(3):
    ax.plot(unv(cmm[i]),iod_i[mins][prog[i]], '.', label=labels[i])
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.set_xlim(19400, 19700)
ax.set_ylim(20000, 26500)
ax.legend(loc='lower left')

# calculating energy differences
# For associating the correct 'n' , we need to numerate the progressions (find the corrisponding v') first
# This is done manually comparing with Tab 2 & 3, p. 47a, b in Staatsex-Jod2-Molekül.pdf
# As data is saved in the reversed order (increasing lambda, not energy), we need to invert the numbering
nums = [[],[],[]] 
n = {}
for i in range(3):
    dcm[i] = cmm[i][:-1] - cmm[i][1:]
n[0] = np.arange(47, 16, -1) 
n[1] = np.arange(27, 14, -1) 
n[2] = np.arange(20,  8, -1) 
nums[0] = np.arange(46, 16, -1) + 0.5
nums[1] = np.arange(26, 14, -1) + 0.5
nums[2] = np.arange(19,  8, -1) + 0.5


# give out latex tables:
str_l = r"\cline{1-7}\cline{9-11}\cline{13-15}" + "\n"
str_l2 = r"\cline{1-7}\cline{9-11}" + "\n"
str_l3 = r"\cline{1-7}" + "\n"
str_l4 = r"\cline{1-3}" + "\n"
coll_length = 16
if print_tab:
    f1 = open("table1.txt", "w+")
    f1.write("\\begin{tabular}{\n" + "| l| l |l ||c|\n"*3 + "l| l |l |}\n")
    f1.write(str_l)
    for j in [0, 0, 1, 2]:
        string = ("$v'$ & $\\frac{\sigma_{%i,v'}}{\cm}$ & " + \
                "$\\frac{\Delta \sigma_{%i,v'}}{\cm}$")%(j, j) + \
                (["&&"]*2 +  [r"\\"])[j] + "\n"
        f1.write(string)
    f1.write("&"*14 + r" \\" + str_l)
    for i in range(12):
        tab = ("%i & %i & %i & &"*3 + "%i & %i & %i \\\\ \n")\
                %(n[0][i], unv(cmm[0])[i], usd(cmm[0])[i], \
                n[0][i+coll_length], unv(cmm[0])[i+coll_length], usd(cmm[0])[i+coll_length], \
                n[1][i], unv(cmm[1])[i], usd(cmm[1])[i], \
                n[2][i], unv(cmm[2])[i], usd(cmm[2])[i])
        f1.write(tab + str_l)
    for i in [12]:
        tab = ("%i & %i & %i & &"*2 + "%i & %i & %i \\\\\n")\
                %(n[0][i], unv(cmm[0])[i], usd(cmm[0])[i], \
                n[0][i+coll_length], unv(cmm[0])[i+coll_length], usd(cmm[0])[i+coll_length], \
                n[1][i], unv(cmm[1])[i], usd(cmm[1])[i])
        f1.write(tab + str_l2)
    for i in range(13, coll_length -1):
        tab = ("%i & %i & %i & &" + "%i & %i & %i \\\\\n")\
                %(n[0][i], unv(cmm[0])[i], usd(cmm[0])[i], \
                n[0][i+coll_length], unv(cmm[0])[i+coll_length], usd(cmm[0])[i+coll_length])
        f1.write(tab + str_l3)
    for i in [15]:
        tab = ("%i & %i & %i\n \\\\\n")\
                %(n[0][i], unv(cmm[0])[i], usd(cmm[0])[i])
        f1.write(tab + str_l4)
    f1.write(r"\end{tabular}" + "\n")
# give out latex tables of diferrences
if print_tab_diff:
    print("| l| l |l ||c|\n"*3 + "l| l |l |}")
    print(str_l)
    for j in [0, 0, 1, 2]:
        string =("$v' + \\frac{1}{2}$ & " + \
                "$\\frac{\Delta G_{%i} }{ \\cm} $ & " + \
                "$\\frac{\Delta (\Delta G) }{ \\cm} $")%(j) + (["&&"]*2 +  [r"\\"])[j]
        print(string)
    print("&"*14 + r" \\" + str_l)
    for i in range(11):
        tab = ("%.1f & %i & %i & & "*3 + "%.1f & %i & %i \\\\")\
                %(nums[0][i], unv(dcm[0])[i], usd(dcm[0])[i], \
                nums[0][i+15], unv(dcm[0])[i+15], usd(dcm[0])[i+15], \
                nums[1][i], unv(dcm[1])[i], usd(dcm[1])[i], \
                nums[2][i], unv(dcm[2])[i], usd(dcm[2])[i])
        print(tab, str_l)
    for i in [11]:
        tab = ("%.1f & %i & %i & & "*2 + "%.1f & %i & %i \\\\")\
                %(n[0][i], unv(dcm[0])[i], usd(dcm[0])[i], \
                nums[0][i+15], unv(dcm[0])[i+15], usd(dcm[0])[i+15], \
                nums[1][i], unv(dcm[1])[i], usd(dcm[1])[i])
        print(tab, str_l2)
    for i in range(12, 14):
        tab = ("%.1f & %i & %i & & " + "%.1f & %i & %i \\\\")\
                %(nums[0][i], unv(dcm[0])[i], usd(dcm[0])[i], \
                nums[0][i+15], unv(dcm[0])[i+15], usd(dcm[0])[i+15])
        print(tab, str_l3)
    for i in [14]:
        tab = ("%.1f & %i & %i \\\\")\
                %(nums[0][i], unv(dcm[0])[i], usd(dcm[0])[i])
        print(tab, str_l4)

# save all arrays
array_dir = "arrays/"
str_array_all = ["cmm", "dcm", "nums"]
for str_array in str_array_all:
    np.save(array_dir + str_array, eval(str_array))

if plot_show:
    fig0.show()
    fig1.show()
    fig2.show()
    fig3.show()
if save_fig: 
    fig0.savefig(fig_dir + "absorp_03.pdf")
    fig1.savefig(fig_dir + "absorp_03_detail_01.pdf")
    fig2.savefig(fig_dir + "absorp_03_detail_02.pdf")
    fig3.savefig(fig_dir + "absorp_03_detail_03.pdf")
