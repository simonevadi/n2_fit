import os
import numpy as np
from n2_fit import N2_fit


nc = N2_fit()

file = 'belchem_n2_136'
spectra = np.loadtxt(os.path.join('data', file+'.txt'))
energy = spectra[:, 0]
intensity = spectra[:, 1]

dict_fit = nc.get_initial_guess(energy, intensity,
                     n_peaks=6, print_initial_guess=False)

dict_fit['voigt6']['amplitude'] = 0.02

nc.fit_n2(energy, intensity,
            title=file+'_modified_initial_guess',
            dict_fit = dict_fit, 
            n_peaks = 6, 
            plot_initial_guess=False, 
            print_fit_results=False, 
            save_results='results', 
            show_results=False)




