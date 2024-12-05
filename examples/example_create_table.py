import numpy as np
from n2_fit import CreateTable

nc = CreateTable()

nc.create_table(savepath='tables/skewedVoigt.csv',
                fwhm_g = None# np.arange(0.01, .15, 0.001), 
                )
nc.plot_fwhm_vs_rp(table_to_plot='../tables/skewedVoigt.csv', save_path='tables/ratio_vs_rp.pdf')
    
