import numpy as np
from n2_fit import CreateTable

nc = CreateTable()

nc.create_table(savepath='tables/skewedVoigt.csv',
                fwhm_g = np.arange(10, 150+1, 1),
                fwhm_l = np.arange(110, 120, .5) 
                )
nc.plot_table(table_to_plot='tables/skewedVoigt.csv', 
              save_path='tables/ratio_vs_rp.pdf', 
              show_plot=True)
    
