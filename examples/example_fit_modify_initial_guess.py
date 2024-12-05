from n2_fit import N2_fit


gamma=0.0565

nc = N2_fit()

dict_fit = nc.get_initial_guess('data/belchem_n2_136.txt', 
                     n_peaks=6,gamma=gamma, print_initial_guess=False)

# dict_fit['voigt6']['amplitude'] = 0.02
# nc.fit_n2('data/belchem_n2_136.txt',
#             dict_fit = dict_fit, 
#             n_peaks = 6, 
#             gamma=gamma, 
#             plot_initial_guess=False, 
#             print_fit_results=False, 
#             save_results='results', 
#             show_results=False)




