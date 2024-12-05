import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from lmfit.models import LinearModel
from lmfit import Parameters

from numpy import pi

from .models import Models
from .helper_functions import calculate_skewed_voigt_amplitude
from .helper_functions import extract_RP_ratio_for_table

class CreateTable:
    """
    A class to fit the N2 spectra and return the RP and the 
    1st valley over 3rd peak ratio
    
    instantiate with 
      from .base import *
      from bessyii.plans.n2fit import N2_fit
      N2fit_class = N2_fit(db)
      fit_n2 = N2fit_class.fit_n2

    then use with:
    
      fit_n2(identifier,...)

    """
    def __init__(self):

        self.theoretical_centers=np.array([400.880,401.114,401.341,
                                           401.563,401.782,401.997,
                                           402.208,402.414])
        self.first_peak = 400.76
        self.theoretical_intensities=np.array([1,0.9750,0.5598,0.2443,
                                            0.0959,0.0329,0.0110,0.0027])
        self.voigt_intensities = np.array([0.17458469651855146,
                                           0.16666979884071098,
                                           0.09333082103718075,
                                           0.04077470824430837,
                                           0.015676096467807606,
                                           0.005823501673555101,
                                            0.002160335806013886])
        self.models = Models()
        
    
    
    def make_model(self, dict_fit, fit_gamma = False):

        pars = Parameters()

        lin_mod = LinearModel(prefix='lin_')
        pars.update(lin_mod.make_params())
        pars['lin_slope'].set(value=dict_fit['linear']['slope'], min=0)
        pars['lin_intercept'].set(value=dict_fit['linear']['intercept'])
        mod = lin_mod

        for voigt_n in list(dict_fit.keys()):
            if 'voigt' in voigt_n:
                voigt_dict = dict_fit[voigt_n]
                fit = self.models.config_SkewedVoigtModel(voigt_n,
                                                voigt_dict['prefix'], 
                                                voigt_dict['center'], 
                                                voigt_dict['center_low_lim'], 
                                                voigt_dict['center_high_lim'], 
                                                voigt_dict['sigma'], 
                                                voigt_dict['sigma_low_lim'], 
                                                voigt_dict['sigma_high_lim'], 
                                                voigt_dict['amplitude'], 
                                                voigt_dict['amplitude_low_lim'], 
                                                voigt_dict['amplitude_high_lim'],
                                                voigt_dict['gamma'], 
                                                voigt_dict['skew_parameter'], 
                                                pars, 
                                                vary_gamma=fit_gamma)
                mod = mod + fit
        
        return mod, pars
    

    def prepare_param_for_table(self, fwhm_g=0.07, gamma=0.0565):

        sigma_g   = fwhm_g/2.35
        fwhm_g    = 2.355*sigma_g
        intercept = 0
        n_peaks   = 7
        lin_slope = 0
        skew_param = 0
        
        guess = {}
        for index in range(1, n_peaks+1):
            vc           = self.theoretical_centers[index-1]
            vc_intensity = self.voigt_intensities[index-1]
            guess[f'vc{index}'] = vc 
            guess[f'amp{index}']  = vc_intensity
        
        dict_fit = {}
        for index, peak_number in enumerate(range(1, n_peaks+1)):
            model_name = f'voigt{peak_number}'
            prefix = f'v{peak_number}_'
            center = guess[f'vc{peak_number}']
            amplitude = guess[f'amp{peak_number}']
            dict_fit[model_name] = {
                'prefix': prefix, 
                'center':center, 
                'center_low_lim':center - fwhm_g, 
                'center_high_lim':center + fwhm_g, 
                'sigma':sigma_g, 'sigma_low_lim':sigma_g-1, 'sigma_high_lim':sigma_g+1,
                'amplitude':amplitude, 'amplitude_low_lim':amplitude-1, 
                'amplitude_high_lim':amplitude+1, 
                'gamma': gamma, 'gamma_low_lim':gamma-1, 'skew_parameter':skew_param
            }
        
        dict_fit['linear'] = {'slope':lin_slope, 'intercept':intercept}
        
        return dict_fit
        
    def create_table(self, savepath, fwhm_g=None, fwhm_l=None):
        if fwhm_g is None:
            fwhm_g = np.arange(0.01, .15, 0.001)
        if fwhm_l is None:
            gamma = np.arange(0.055, 0.0601, 0.0005)
        else:
            gamma = fwhm_l/2/1000
        
        results = []
        total_iterations = len(gamma) * len(fwhm_g)
        progress_bar = tqdm(total=total_iterations, desc="Creating Table")

        for g in gamma:
            for f in fwhm_g:
                dict_fit = self.prepare_param_for_table(fwhm_g=f, gamma=g)
                mod, pars = self.make_model(dict_fit)
                energy = np.arange(self.theoretical_centers[0]-1, self.theoretical_centers[-1]+1, 0.01)
                intensity = mod.eval(pars, x=energy)
                vp_ratio = extract_RP_ratio_for_table(energy, intensity, pars)
                rp = int(self.first_peak / f)
                results.append({'FWHM_l (meV)': np.round(2*g*1000,2), 'FWHM_g (meV)': np.round(f*1000, 2), 'RP': rp, '3P1V Ratio': vp_ratio})
                progress_bar.update(1)  # Update the progress bar per iteration

        progress_bar.close()

        # Convert results to DataFrame and save as CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(savepath, index=False)
        return df_results    

    def plot_fwhm_vs_rp(self, table_to_plot='tables/table.csv', save_path=False, show_plot=False):
        # Read the data from the CSV file
        df = pd.read_csv(table_to_plot)

        # Create a figure and axis for the plot
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Iterate over each unique gamma value and plot
        for gamma in df['FWHM_l (meV)'].unique():
            # Filter the DataFrame by gamma
            gamma_data = df[df['FWHM_l (meV)'] == gamma]
            
            # Plot FWHM vs RP for each gamma
            ax.plot(gamma_data['3P1V Ratio'], gamma_data['RP'], linestyle='-', label=f'FWHM_l: {gamma} meV')

        # Setting plot labels and titles
        ax.set_xlabel('3rdPeak/1st valley ratio', fontsize=12)
        ax.set_ylabel('Resolving Power', fontsize=12)
        ax.set_title('3rdPeak/1st valley ratio vs Resolving Power', fontsize=14)
        ax.legend(title='Lorentian FWHM', fontsize=10, title_fontsize=12)

        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()


   
















