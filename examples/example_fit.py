from n2_fit import N2_fit


gamma=0.0565

nc = N2_fit()

nc.fit_n2_3peaks('data/belchem_n2_short.txt', dict_fit=None, 
               n_peaks = 3,
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2_3peaks('data/N1spistarBL25SU_short.txt', dict_fit=None, 
               n_peaks = 3,
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2_3peaks('data/belchem_n2_136_short.txt', dict_fit=None, 
               n_peaks = 3, 
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/belchem_n2_136.txt', dict_fit=None, 
               n_peaks = 6, 
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/belchem_n2.txt', dict_fit=None, 
               n_peaks = 6, 
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/n2.txt', dict_fit=None, 
               n_peaks = 7, 
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)

nc.fit_n2('data/N1spistarBL25SU.txt', dict_fit=None, 
               n_peaks = 7,
               gamma=gamma, 
               plot_initial_guess=False, 
               print_fit_results=False, 
               save_results='results', 
               show_results=False)


