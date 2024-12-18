import os
import numpy as np
import pandas as pd
from n2_fit import N2_fit


fwhm_l=113
nc = N2_fit()

files_list = [138, 139, 140, 141, 142, 143, 144, 145,
              146, 147, 171, 172, 173, 174, 175, 177,
              178, 179, 180, 183, 184]

for file_n in files_list:
    file = f'00{file_n}_primary'
    spectra = pd.read_csv(os.path.join('data', file+'.csv'))
    energy = spectra['pgm_en'].to_numpy()
    intensity = spectra['kth01'].to_numpy()
    nc.fit_n2(energy, intensity, title=file, 
            dict_fit=None, n_peaks=6, 
            fwhm_l=fwhm_l, 
            plot_initial_guess=False, 
            print_fit_results=False, 
            save_results='results', 
            show_results=False)
