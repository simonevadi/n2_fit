import os
import numpy as np
from n2_fit import N2_fit


nc = N2_fit()
files_list = ['belchem_n2_short', 'N1spistarBL25SU_short', 'belchem_n2_136_short']
for file in files_list:
    spectra = np.loadtxt(os.path.join('data', file+'.txt'))
    energy = spectra[:, 0]
    intensity = spectra[:, 1]
    nc.fit_n2_3peaks(energy, intensity, 
                    title=file, dict_fit=None, 
                    fwhm_l=114, 
                    plot_initial_guess=False, 
                    print_fit_results=False, 
                    save_results='results', 
                    show_results=False)



