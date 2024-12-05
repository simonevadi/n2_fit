from n2_fit import N2_fit

if __name__ == '__main__':
    db=None
    nc = N2_fit(db)
    # energy, intensity  = n_class.retrieve_spectra('belchem_n2.txt')
    # dict_fit = n_class.get_initial_guess('belchem_n2.txt', n_peaks=5)
    
    # model, parameters = n_class.make_model('belchem_n2.txt', dict_fit)
    
    # n_class.plot_initial_guess(energy, intensity, model, parameters)

    # out = n_class._fit_n2('belchem_n2.txt', dict_fit, print_fit_results=False)
    # fwhm_l, fwhm_g, rp, ratio, vc1 = n_class.analyze_fit_results(energy, intensity, out, n_peaks=3)
    # n_class._print_fit_results(fwhm_l, fwhm_g, rp, ratio, vc1)
    # n_class.plot_fit(energy, intensity, out)

    # nc.fit_n2('belchem_n2.txt', dict_fit=None, 
    #                n_peaks = 6, 
    #                plot_initial_guess=False, 
    #                print_fit_results=False, 
    #                save_results=True, 
    #                show_results=False)
    
    # # dict_fit = n_class.get_initial_guess('n2.txt', n_peaks=7)

    # nc.fit_n2('n2.txt', dict_fit=None, 
    #                n_peaks = 7, 
    #                plot_initial_guess=False, 
    #                print_fit_results=False, 
    #                save_results=True, 
    #                show_results=False)
    nc.create_table()
    nc.plot_fwhm_vs_rp()
    
