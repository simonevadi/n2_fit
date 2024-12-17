import os
import numpy as np
from n2_fit import N2_fit


fwhm_l=113
nc = N2_fit()

file = 'belchem_n2_136'
spectra = np.loadtxt(os.path.join('data', file+'.txt'))
energy = spectra[:, 0]
intensity = spectra[:, 1]
nc.fit_n2(energy, intensity, title=file, 
          dict_fit=None, n_peaks=6, 
          fwhm_l=fwhm_l, 
          plot_initial_guess=False, 
          print_fit_results=True, 
          save_results='results', 
          show_results=False)

file = 'belchem_n2'
spectra = np.loadtxt(os.path.join('data', file+'.txt'))
energy = spectra[:, 0]
intensity = spectra[:, 1]
nc.fit_n2(energy, intensity, title=file, 
          dict_fit=None, n_peaks=5, 
          fwhm_l=fwhm_l, 
          plot_initial_guess=False, 
          print_fit_results=True, 
          save_results='results', 
          show_results=False)

file = 'n2'
spectra = np.loadtxt(os.path.join('data', file+'.txt'))
energy = spectra[:, 0]
intensity = spectra[:, 1]
nc.fit_n2(energy, intensity, title=file, 
          dict_fit=None, n_peaks=5, 
          fwhm_l=fwhm_l, 
          plot_initial_guess=False, 
          print_fit_results=True, 
          save_results='results', 
          show_results=False)

file = 'N1spistarBL25SU'
spectra = np.loadtxt(os.path.join('data', file+'.txt'))
energy = spectra[:, 0]
intensity = spectra[:, 1]
nc.fit_n2(energy, intensity, title=file, 
          dict_fit=None, n_peaks=7, 
          fwhm_l=fwhm_l, 
          plot_initial_guess=False, 
          print_fit_results=True, 
          save_results='results', 
          show_results=False)

