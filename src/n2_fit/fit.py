import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json

from lmfit.models import LinearModel
from lmfit import Parameters

from numpy import pi

from .models import Models
from .helper_functions import remove_neg_values, calculate_skewed_voigt_amplitude
from .helper_functions import find_first_max, find_max_around_theoretical
from .helper_functions import extract_bandwidth_and_rp, extract_RP_ratio
from .helper_functions import find_closest_fwhm, format_val, convert_to_json_serializable
from .helper_functions import calculate_rms

class N2_fit:
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
    def __init__(self, db=None):
        self._db = db
        self.tiny = 1.0e-15
        self.s2   = np.sqrt(2)
        self.s2pi = np.sqrt(2*pi)
        self.theoretical_centers=np.array([400.880,401.114,401.341,
                                           401.563,401.782,401.997,
                                           402.208,402.414])
        self.first_peak = 400.76
        self.theoretical_intensities=np.array([1,0.9750,0.5598,0.2443,
                                            0.0959,0.0329,0.0110,0.0027])
        self.models = Models()
        
    def retrieve_spectra(self, identifier, motor=None, detector=None):
        """
        Retrieve the motor and detector values from one scan

        Parameters
        ----------
        identifier : negative int or string
            for the last scan -1
            or use the db indentifier
            'Jens': if available it loads the data from P04,
            the old beamline of J.Viefhaus
        motor : string
            the motor and axis name connected by a _
            for instance m1.tx would be m1_tx
        detector : string
            the detector to retrieve, if more than one detector was used
            in the scan and we don't want to use the first one

        Return
        --------
        x,y : np.array
            two arrays containing motor and detector values
        """
        if '.' in identifier:
            dat       = np.loadtxt(identifier)
            x    = dat[:, 0]
            y = dat[:, 1]
        else:
            run       = self._db[identifier]
            if detector == None:
                detector  = run.metadata['start']['detectors'][0]
            if motor == None:
                motor = run.metadata['start']['motors'][0]
            spectra   = run.primary.read()
            x    = np.array(spectra[motor])
            y = np.array(spectra[detector])

            x,y = remove_neg_values(x,y)
        return x, y
    
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
    
    def _fit_n2(self, energy, intensity, dict_fit, fit_gamma=False, print_fit_results=False):
        """
        
        """
        norm = np.max(intensity)
        intensity = intensity/norm

        mod, pars = self.make_model(dict_fit, fit_gamma=fit_gamma)

            
        out = mod.fit(intensity, pars, x=energy)
        delta = out.eval_uncertainty(x=energy)
        
        if print_fit_results == True:
            print(out.fit_report(min_correl=0.5))

        return out
    
    def plot_initial_guess(self, energy, intensity, mod, pars):
        norm = np.max(intensity)
        intensity = intensity/norm
        energy_plot = np.arange(energy[0], energy[-1], .001)
        init = mod.eval(pars, x=energy_plot)

        plt.rc("font", size=12,family='serif')
        fig, axes = plt.subplots(1, 1, figsize=(8.0, 16.0))
        axes.plot(energy_plot, init, 'orange' ,label='initial guess')
        axes.scatter(energy, intensity, label='data')
        plt.show()
        

    def plot_fit(self, title, energy, intensity, out, fit_results, save_results=False, show_results=True):
        norm = np.max(intensity)
        intensity = intensity / norm

        energy_plot = np.arange(energy[0], energy[-1], .001)
        plt.rc("font", size=12, family='serif')
        fig, axs = plt.subplots(3, 1, figsize=(8.0, 16.0), gridspec_kw={'height_ratios': [3, .5, .5]})
        plt.suptitle(title)

        ax = axs[0]  # Upper plot for the fit and data
        # Data
        ax.scatter(energy, intensity, label='Data', s=40)
        # Plot initial guesses 
        ax.plot(energy, out.init_fit, color='orange', alpha=1, linewidth=1.5, label='Initial Guess')

        # Centers
        for i in range(1, 20):
            try:
                ax.axvline(x=out.params[f'v{i}_center'].value, color='grey', linewidth=0.1, linestyle='--')
            except Exception as e:
                continue

        # Fit
        ax.plot(energy_plot, out.eval(x=energy_plot), 'r', label=f'Fit, Lorentzian: {np.round(out.params["v1_gamma"].value * 2 * 1000, 2)} meV')

        # Voigt Components
        comps = out.eval_components(x=energy_plot)
        for i in range(1, 20):
            try:
                center = np.round(out.params[f'v{i}_center'].value, 2)
                intensity_component = np.round(out.params[f'v{i}_amplitude'].value, 2)
                ax.plot(energy_plot, comps[f'v{i}_'] + comps['lin_'], '--', label=f'Voigt{i}: {center} meV, Intensity: {intensity_component}')
            except:
                continue

        # Linear Component
        ax.plot(energy_plot, comps['lin_'], '--', label='Linear component')


        

        legend = ax.legend()
        # summary text
        summary_text = (
            f"{'Lorentzian [meV] '}:   {format_val(fit_results['fwhm_l']):>10}\n"
            f"{'-' * 50}\n"
            f"{'FWHM_g [meV]    '}:  {format_val(fit_results['fwhm_g'])}\n"
            f"{'1st Peak [eV]        '}: {format_val(fit_results['vc1'])}\n"
            f"{'Energy Shift [eV]  '}:   {format_val(fit_results['energy_shift'])}\n"
            f"{'Valley/peak Ratio  '}:   {format_val(fit_results['ratio'])}\n"
            f"{'RP from fit            '}:  {format_val(fit_results['rp_from_fit'])}\n"
            f"{'RP from v/p ratio  '}:  {format_val(fit_results['rp_from_table'])}"
        )
        # Placing the text just below the legend by calculating the legend's bbox coordinates
        legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        
        ax.text(legend_box.x0+0.08, legend_box.y0-0.02,
                summary_text, transform=fig.transFigure, 
                fontsize=12,  
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Photon energy [eV]')
        ax.set_ylabel('Normalized Intensity [a.u.]')

        # Subplot for residuals
        ax2 = axs[1]  # Lower plot for residuals
        ax2.plot(energy, fit_results['residuals'], 'k.', label=f'Residuals, RMS: {np.round(fit_results["rms"],3)}')
        ax2.set_xlabel('Photon energy [eV]')
        ax2.set_ylabel('Residuals')
        ax2.legend()

        # Plotting histogram of residuals
        ax3 = axs[2]
        fitted_values = out.eval(x=energy)
        ax3.scatter(fitted_values, fit_results['residuals'], color='blue', alpha=0.5)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('Residuals')
        ax3.title.set_text('Residuals vs. Fitted Values')
        # Place text in the top right corner of the ax3 subplot
        ax3.text(0.95, 0.9, 'For a good fit there is no correlation',
                verticalalignment='top', horizontalalignment='right',
                transform=ax3.transAxes,
                fontsize=12, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))


        plt.tight_layout()  # Adjust layout to make room for the suptitle

        if save_results:
            save_path = os.path.join(save_results, f'{title}_lor{fit_results["fwhm_l"]}.pdf')
            plt.savefig(save_path)
        if show_results:
            plt.show()


    def get_initial_guess(self,energy,intensity,n_peaks=5, gamma=0.0565, energy_first_peak='auto'):
        # normalize intensity
        norm = np.max(intensity)
        intensity = intensity/norm
        sigma_g     = 0.01
        amp_mf      = 1.3  #scaling value for minimal amplitude
        gamma_min   = 0
        fwhm_g      = 2.355*sigma_g
        lin_slope   = 0
        sigma_g_min = 0
        sigma_g_max = np.inf
        center_scale_factor = 2
        intercept = np.average(intensity[0:3])
        differences = np.diff(self.theoretical_centers)

        guess = {}
        for index in range(1, n_peaks+1):
            if energy_first_peak == 'auto' and index ==1:
                vc, vc_intensity, argmax = find_first_max(energy,intensity, fwhm_g)
                vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-intercept, vc, fwhm_g*4)
                vc_intensity = calculate_skewed_voigt_amplitude(vc, sigma_g, gamma, 0, vc_intensity)
            else:
                vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-intercept, vc+differences[index-1], fwhm_g*4)
                vc_intensity = calculate_skewed_voigt_amplitude(vc, sigma_g, gamma, 0, vc_intensity)

            
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
                'center_low_lim':center - fwhm_g/center_scale_factor, 
                'center_high_lim':center + fwhm_g/center_scale_factor, 
                'sigma':sigma_g, 'sigma_low_lim':sigma_g_min, 'sigma_high_lim':sigma_g_max,
                'amplitude':amplitude, 'amplitude_low_lim':amplitude / amp_mf, 
                'amplitude_high_lim':amplitude*amp_mf, 
                'gamma': gamma, 'gamma_low_lim':gamma_min, 'skew_parameter':0.0
            }
        
        # guess the slope
        initial_avg_x = np.mean(energy[:3])
        initial_avg_y = np.mean(intensity[:3])
        final_avg_x = np.mean(energy[-3:])
        final_avg_y = np.mean(intensity[-3:])
        lin_slope = (final_avg_y - initial_avg_y) / (final_avg_x - initial_avg_x)
        dict_fit['linear'] = {'slope':lin_slope, 'intercept':intercept}
        
        return dict_fit

    def analyze_fit_results(self, energy, intensity, fit, n_peaks):
        fit_results = {}
        
        gamma = fit.params['v1_gamma'].value
        fit_results['fwhm_l'] = np.round(2*gamma*1000,2)
        
        fwhm_g, rp = extract_bandwidth_and_rp(fit, n_peaks=n_peaks)
        fit_results['fwhm_g'] = fwhm_g
        fit_results['rp_from_fit'] = rp

        fit_results['ratio'] = extract_RP_ratio(energy, intensity, fit)
        fit_results['vc1'] = fit.params['v1_center'].value
        fit_results['energy_shift'] = fit_results['vc1'] - self.first_peak
        
        fwhm_g_rp = find_closest_fwhm('../tables/skewedVoigt.csv',
                                     fit_results['ratio'],
                                     fit_results['fwhm_l'])
        
        fit_results['rp_from_table'] = int(self.first_peak/(fwhm_g_rp/1000)) 
        
        fit_results['residuals'] = intensity - fit.best_fit
        fit_results['rms'] = calculate_rms(fit_results['residuals'])

        return fit_results
        


    def fit_n2(self, scan, dict_fit=None, n_peaks=5, 
               plot_initial_guess=False, print_fit_results=False, 
               save_results=False, show_results=True, gamma=0.057):
        print(f'Starting the fit for {scan}')
        if save_results:
            if not os.path.exists(save_results):
                os.makedirs(save_results)
        energy, intensity  = self.retrieve_spectra(scan)
        if dict_fit is None:
            dict_fit = self.get_initial_guess(energy,intensity, n_peaks, gamma=gamma)
        
        model, parameters = self.make_model(dict_fit)
        
        if plot_initial_guess:
            self.plot_initial_guess(energy, intensity, model, parameters)

        out = self._fit_n2(energy,intensity, dict_fit, print_fit_results=print_fit_results)
        fit_results = self.analyze_fit_results(energy, 
                                                intensity, 
                                                out,
                                                n_peaks=n_peaks)
        base_name = os.path.basename(scan)
        title = os.path.splitext(base_name)[0]
        if save_results:
            json_ready_results = convert_to_json_serializable(fit_results)
            analysis_save_path = os.path.join(save_results, f'{title}.json')
            with open(analysis_save_path, 'w') as json_file:
                json.dump(json_ready_results, json_file, indent=4)

        self._print_fit_results(fit_results)
        self.plot_fit(title, energy, intensity, out, fit_results, save_results=save_results, show_results=show_results)
        
    def _print_fit_results(self, fit_results):
        """
        Print the fitting results, including resolving power and peak ratios, in a tabular format using string formatting.
        Handles cases where minimum and maximum values might not be provided.

        Parameters:
        - sigma: Gaussian sigma from the fit.
        - gamma: Lorentzian gamma from the fit.
        - sigma_min: Minimum Gaussian sigma used in fitting, if provided.
        - sigma_max: Maximum Gaussian sigma used in fitting, if provided.
        - gamma_min: Minimum Lorentzian gamma used in fitting, if provided.
        - gamma_max: Maximum Lorentzian gamma used in fitting, if provided.
        - center: Center of the first peak.
        - vp_ratio: Valley/peak ratio from the fit.
        - vp_ratio_min: Minimum valley/peak ratio, if computed.
        - vp_ratio_max: Maximum valley/peak ratio, if computed.
        """
 
        print('Results Summary:\n')
        print(f"{'Lorentzian [meV]':<20}{format_val(fit_results['fwhm_l']):>10}")
        print('-' * 50)
        print(f"{'FWHM_g [meV]':<20}{format_val(fit_results['fwhm_g']):>10}")
        print(f"{'1st Peak [eV]':<20}{format_val(fit_results['vc1']):>10}")
        print(f"{'Energy Shift [eV]':<20}{format_val(fit_results['energy_shift']):>10}")
        print(f"{'Valley/peak Ratio':<20}{format_val(fit_results['ratio']):>10}")
        print(f"{'RP from fit':<20}{format_val(fit_results['rp_from_fit']):>10}")
        print(f"{'RP from v/p ratio':<20}{format_val(fit_results['rp_from_table']):>10}\n\n")




















