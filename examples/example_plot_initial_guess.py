import os
import numpy as np
import pandas as pd
from n2_fit import N2_fit


fwhm_l=113
nc = N2_fit()

file = '00138_primary'
spectra = pd.read_csv(os.path.join('data', file+'.csv'))
energy = spectra['pgm_en'].to_numpy()
intensity = spectra['kth01'].to_numpy()
nc.fit_n2(energy, intensity, title=file, 
          dict_fit=None, n_peaks=6, 
          fwhm_l=fwhm_l, 
          plot_initial_guess=True, 
          print_fit_results=False, 
          save_results='results', 
          show_results=False)

