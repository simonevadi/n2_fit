from n2_fit import N2_fit


fwhm_l=113
nc = N2_fit()

nc.fit_n2('data/belchem_n2_136.txt', dict_fit=None, 
               n_peaks = 6, 
               fwhm_l=fwhm_l, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/belchem_n2.txt', dict_fit=None, 
               n_peaks = 6, 
               fwhm_l=fwhm_l, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/n2.txt', dict_fit=None, 
               n_peaks = 7, 
               fwhm_l=fwhm_l, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/N1spistarBL25SU.txt', dict_fit=None, 
               n_peaks = 7,
               fwhm_l=fwhm_l, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)


