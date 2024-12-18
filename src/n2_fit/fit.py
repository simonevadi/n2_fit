import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pprint import pprint


from .models import N2SkewedVoigtModel
from .helper_functions import remove_neg_values
from .helper_functions import extract_bandwidth_and_rp, extract_RP_ratio
from .helper_functions import find_closest_fwhm, format_val, convert_to_json_serializable
from .helper_functions import calculate_rms, clean_data

from .n2_peaks_parameters import first_peak

class N2_fit:
    """
    A class dedicated to fitting N2 spectra, calculating resolving power (RP), and analyzing the ratio between the 
    first valley and the third peak in spectral data.

    Attributes:
        _db (database connection or None): Database connection if provided, used to retrieve spectra data.
        tiny (float): A small number to avoid division by zero in calculations.
        s2 (float): Square root of 2, used in various calculations.
        s2pi (float): Square root of 2 times pi, used in normalization.
        theoretical_centers (np.array): Theoretical center positions for spectral peaks.
        first_peak (float): The energy position of the first peak, used as a reference for calculations.
        theoretical_intensities (np.array): Theoretical intensities of peaks at the theoretical centers.
        model (object): An instance of a modeling class (e.g., N2SkewedVoigtModel) for spectral fitting.
    """
    def __init__(self, db=None):
        """
        Initializes the N2_fit class.

        Args:
            db (optional): A database connection to retrieve scan data. Defaults to None.

        Attributes:
            _db (database connection or None): Database connection if provided.
            tiny (float): A small number to avoid division by zero in calculations.
            s2 (float): Square root of 2, used in various calculations.
            s2pi (float): Square root of 2 times pi, used in normalization.
            theoretical_centers (np.array): Theoretical center positions for spectral peaks.
            first_peak (float): The energy position of the first peak, used as a reference for calculations.
            theoretical_intensities (np.array): Theoretical intensities of peaks at the theoretical centers.
            model (object): An instance of a modeling class (e.g., N2SkewedVoigtModel) for spectral fitting.
        """
        self._db = db
        self.tiny = 1.0e-15
        self.s2   = np.sqrt(2)
        self.s2pi = np.sqrt(2*np.pi)
    
    def _fit_n2(self, energy, intensity, dict_fit, print_fit_results=False):
        """
        Performs fitting on N2 spectra using the specified model and parameters.

        Args:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of intensity values for the spectral data.
            dict_fit (dict): Dictionary containing fit parameters and model configurations.
            print_fit_results (bool, optional): Flag to indicate whether to print the fit results. Defaults to False.

        Returns:
            ModelResult: The result of the fit containing fit parameters and statistical data.
        """

        mod, pars = self.model.make_model(dict_fit)

            
        fit = mod.fit(intensity, pars, x=energy)
        
        if print_fit_results == True:
            print(fit.fit_report(min_correl=0.5))

        return fit
    
    def _plot_initial_guess(self, energy, intensity, mod, pars, norm=1):
        """
        Plots the spectral fitting results including the original data, the fit, initial guesses, and residuals.

        Args:
            title (str): Title of the plot, usually derived from the data set name.
            energy (np.array): The energy values corresponding to the spectral data.
            intensity (np.array): The intensity values of the spectral data.
            out (ModelResult): The fitting result object from lmfit.
            fit_results (dict): Dictionary containing additional fit results like residuals and metrics.
            intensity_norm (float, optional): Normalization factor for intensity. Defaults to 1.
            save_results (bool or str, optional): If a string is provided, it specifies the path where the plot will be saved. Defaults to False.
            show_results (bool, optional): If True, displays the plot interactively. Defaults to True.
            close_plot (bool, optional): If True, closes the plot after displaying. Defaults to False.

        Displays:
            A multi-panel plot with the top panel showing the data and fit, a middle panel showing residuals,
            and a bottom panel showing residuals vs. fitted values.
        """
        energy_plot = np.arange(energy[0], energy[-1], .001)
        init = mod.eval(pars, x=energy_plot)

        plt.rc("font", size=12,family='serif')
        fig, axes = plt.subplots(1, 1, figsize=(8.0, 16.0))
        axes.plot(energy_plot, init*norm, 'orange' ,label='initial guess')
        axes.scatter(energy, intensity*norm, label='data')

        # Plotting individual Voigt components
        energy_v = energy_plot
        components = mod.eval_components(params=pars,x=energy_v)
        for name, comp in components.items():
            if 'v' in name:  # This condition depends on how the components are named in the model
                axes.plot(energy_v, comp*norm, '--', label=f'{name} component')
        axes.legend()
        plt.show()
        
    def _plot_fit(self, title, energy, intensity, out, fit_results, intensity_norm=1, save_results=False, show_results=True, close_plot=False):
        """
        Plots the spectral fitting results including the original data, the fit, initial guesses, and residuals.

        Parameters:
            title (str): Title of the plot, usually derived from the data set name.
            energy (np.array): The energy values corresponding to the spectral data.
            intensity (np.array): The intensity values of the spectral data.
            out (ModelResult): The fitting result object from lmfit.
            fit_results (dict): Dictionary containing additional fit results like residuals and metrics.
            save_results (bool or str): If a string is provided, it specifies the path where the plot will be saved.
            show_results (bool): Flag to display the plot interactively.

        Displays:
            A multi-panel plot with the top panel showing the data and fit, a middle panel showing residuals,
            and a bottom panel showing residuals vs. fitted values.
        """

        energy_plot = np.arange(energy[0], energy[-1], .001)
        plt.rc("font", size=12, family='serif')
        fig, axs = plt.subplots(3, 1, figsize=(20.0, 40.0), gridspec_kw={'height_ratios': [3, .5, .5]})
        plt.suptitle(title)

        ax = axs[0]  # Upper plot for the fit and data
        # Data
        ax.scatter(energy, intensity*intensity_norm, label='Data', s=40)
        # Plot initial guesses 
        ax.plot(energy, out.init_fit*intensity_norm, color='orange', alpha=1, linewidth=1.5, label='Initial Guess')

        # Centers
        for i in range(1, 20):
            try:
                ax.axvline(x=out.params[f'v{i}_center'].value, color='grey', linewidth=0.1, linestyle='--')
            except Exception as e:
                continue

        # Fit
        ax.plot(energy_plot, out.eval(x=energy_plot)*intensity_norm, 'r', label=f'Fit, Lorentzian: {np.round(out.params["v1_gamma"].value * 2 * 1000, 2)} meV')

        # Voigt Components
        comps = out.eval_components(x=energy_plot)
        for i in range(1, 20):
            try:
                center = np.round(out.params[f'v{i}_center'].value, 2)
                intensity_component = np.round(out.params[f'v{i}_amplitude'].value, 2)
                ax.plot(energy_plot, (comps[f'v{i}_'] + comps['lin_'])*intensity_norm, '--', label=f'Voigt{i}: {center} meV, Intensity: {intensity_component}')
            except:
                continue

        # Linear Component
        ax.plot(energy_plot, comps['lin_']*intensity_norm, '--', label='Linear component')

        # valley and peak positions
        ax.scatter(fit_results['vp_positions']['valley'][0], 
                   fit_results['vp_positions']['valley'][1]*intensity_norm, c='red')
        ax.scatter(fit_results['vp_positions']['peak'][0], 
                   fit_results['vp_positions']['peak'][1]*intensity_norm, c='red')


        

        legend = ax.legend(loc='upper right')
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

        # Plotting residuals
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
            save_path = os.path.join(save_results, f'{title}_lor{fit_results["fwhm_l"]}.png')
            plt.savefig(save_path)
        if show_results:
            plt.show()
        if close_plot or not show_results:
            plt.close()

    def _analyze_fit_results(self, energy, intensity, fit, n_peaks):
        """
        Analyzes the fitting results to extract key performance metrics such as resolving power and peak ratios.

        Args:
            energy (np.array): Array of energy values used in the fit.
            intensity (np.array): Array of measured intensity values used in the fit.
            fit (ModelResult): The fitting result object from lmfit.
            n_peaks (int): The number of peaks considered in the fit.

        Returns:
            dict: A dictionary containing the calculated metrics such as Lorentzian width, Gaussian width, 
                resolving power from fit, and ratios of valley to peak intensities.
        """
        fit_results = {}
        
        gamma = fit.params['v1_gamma'].value
        fit_results['fwhm_l'] = np.round(2*gamma*1000,2)
        
        fwhm_g, rp = extract_bandwidth_and_rp(fit, n_peaks=n_peaks)
        fit_results['fwhm_g'] = fwhm_g
        fit_results['rp_from_fit'] = rp

        fit_results['ratio'], fit_results['vp_positions'] = extract_RP_ratio(energy, intensity, fit)
        fit_results['vc1'] = fit.params['v1_center'].value
        fit_results['energy_shift'] = fit_results['vc1'] - first_peak
        script_dir = os.path.dirname(__file__)
        two_levels_up = os.path.dirname(os.path.dirname(script_dir))
        table_path = os.path.join(two_levels_up, 'tables/skewedVoigt.csv')
        fwhm_g_rp = find_closest_fwhm(table_path,
                                     fit_results['ratio'],
                                     fit_results['fwhm_l'])
        
        
        fit_results['rp_from_table'] = int(first_peak/(fwhm_g_rp/1000)) 
        
        fit_results['residuals'] = intensity - fit.best_fit
        fit_results['rms'] = calculate_rms(fit_results['residuals'])

        return fit_results
        
    def get_initial_guess(self, energy, intensity, n_peaks=5,gamma=0.057, print_initial_guess=False, 
                          model:str='SkewedVoigt', background_model:str='offset'):
        """
        Generates the initial guess parameters for fitting N2 spectra.

        Args:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of intensity values for the spectral data.
            n_peaks (int, optional): Number of peaks to include in the initial guess. Defaults to 5.
            gamma (float, optional): Gamma value for fitting. Defaults to 0.057.
            print_initial_guess (bool, optional): If True, prints the initial guess parameters. Defaults to False.
            model (str, optional): The name of the model to use for fitting. Defaults to 'SkewedVoigt'.
            background_model (str, optional): The type of background model to use. Defaults to 'offset'.

        Returns:
            dict: A dictionary containing the initial guess parameters for the fit.
        """

        if model == 'SkewedVoigt':
            self.model = N2SkewedVoigtModel(background=background_model, 
                                            fit_gamma=False)
        else:
            raise NotImplemented('No other Model is Implemented')
        
        dict_fit = self.model.get_initial_guess(energy,intensity,
                                        n_peaks, gamma=gamma)
        if print_initial_guess:
            pprint(dict_fit)

        return dict_fit

    def fit_n2(self, energy, intensity, title=None, dict_fit=None, n_peaks=5, 
               plot_initial_guess=False, print_fit_results=False, 
               save_results=False, show_results=True, fwhm_l:float=114, 
               model:str='SkewedVoigt', background_model:str='linear', motor=None, detector=None, 
               summary=True, close_plot=False):
        """
        Orchestrates the fitting process for N2 spectral data, including retrieving data, performing the fit,
        analyzing results, and plotting.

        Args:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of intensity values for the spectral data.
            title (str, optional): Title for the plot and saved results. Defaults to None.
            dict_fit (dict, optional): Predefined fitting parameters. If None, initial guesses are used. Defaults to None.
            n_peaks (int, optional): Number of peaks to fit. Defaults to 5.
            plot_initial_guess (bool, optional): If True, plots the initial guesses. Defaults to False.
            print_fit_results (bool, optional): If True, prints the fit report. Defaults to False.
            save_results (bool or str, optional): If True or a path is provided, saves the fit results to a JSON file. Defaults to False.
            show_results (bool, optional): If True, displays the plot. Defaults to True.
            fwhm_l (float, optional): Full width at half maximum (Lorentzian) value. Defaults to 114.
            model (str, optional): The name of the model to use for fitting. Defaults to 'SkewedVoigt'.
            background_model (str, optional): The type of background model to use. Defaults to 'linear'.
            motor (str, optional): Name of the motor to retrieve data from. Defaults to None.
            detector (str, optional): Name of the detector to retrieve data from. Defaults to None.
            summary (bool, optional): If True, prints the fit results summary. Defaults to True.
            close_plot (bool, optional): If True, closes the plot after displaying. Defaults to False.

        Returns:
            tuple: A tuple containing the fit results dictionary and the fit object.
        """
        if model == 'SkewedVoigt':
            self.model = N2SkewedVoigtModel(background=background_model, 
                                            fit_gamma=False)
        else:
            raise NotImplemented('No other Model is Implemented')
        

        title = 'NotNamedData' if title is None else title

        print(f'Starting the fit: {title}')
        gamma = fwhm_l/2000
        if save_results:
            if not os.path.exists(save_results):
                os.makedirs(save_results)

            
        energy, intensity = clean_data(energy, intensity)
        intensity_norm = np.max(intensity)
        intensity /= intensity_norm
        if background_model == 'offset':
            intensity = intensity-np.min(intensity)
        if dict_fit is None:
            dict_fit = self.model.get_initial_guess(energy,intensity,
                                                    n_peaks, gamma=gamma)        
        model, parameters = self.model.make_model(dict_fit)
        
        if plot_initial_guess:
            self._plot_initial_guess(energy, intensity, model, parameters, norm=intensity_norm)
            return

        fit = self._fit_n2(energy,intensity, dict_fit, print_fit_results=print_fit_results)
        
        fit_results = self._analyze_fit_results(energy, 
                                                intensity, 
                                                fit,
                                                n_peaks=n_peaks)

        if save_results:
            json_ready_results = convert_to_json_serializable(fit_results)
            analysis_save_path = os.path.join(save_results, f'{title}.json')
            with open(analysis_save_path, 'w') as json_file:
                json.dump(json_ready_results, json_file, indent=4)

        if summary:
            self._print_fit_results(fit_results)
        self._plot_fit(title, energy, intensity, fit, fit_results, intensity_norm=intensity_norm,
                      save_results=save_results, show_results=show_results,
                      close_plot=close_plot)
        return fit_results, fit
    
    def fit_n2_3peaks(self, energy, intensity, title=None, dict_fit=None, 
               plot_initial_guess=False, print_fit_results=False, 
               save_results=False, show_results=True, fwhm_l:float=114, 
               motor=None, detector=None, summary=False):
        """
        Fits the N2 spectral data with 3 peaks using the fit_n2 method.

        Args:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of intensity values for the spectral data.
            title (str, optional): Title for the plot and saved results. Defaults to None.
            dict_fit (dict, optional): Predefined fitting parameters. If None, initial guesses are used. Defaults to None.
            plot_initial_guess (bool, optional): If True, plots the initial guesses. Defaults to False.
            print_fit_results (bool, optional): If True, prints the fit report. Defaults to False.
            save_results (bool or str, optional): If True or a path is provided, saves the fit results to a JSON file. Defaults to False.
            show_results (bool, optional): If True, displays the plot. Defaults to True.
            fwhm_l (float, optional): Full width at half maximum (Lorentzian) value. Defaults to 114.
            motor (str, optional): Name of the motor to retrieve data from. Defaults to None.
            detector (str, optional): Name of the detector to retrieve data from. Defaults to None.
            summary (bool, optional): If True, prints the fit results summary. Defaults to False.

        Returns:
            tuple: A tuple containing the fit results dictionary and the fit object.
        """
        fit_results, fit = self.fit_n2(energy, intensity, n_peaks=3, dict_fit=dict_fit, 
                    plot_initial_guess=plot_initial_guess, 
                    print_fit_results=print_fit_results, 
                    save_results=save_results, show_results=show_results, 
                    fwhm_l=fwhm_l, model='SkewedVoigt', background_model='offset', 
                    motor=motor, detector=detector, summary=summary,title=title)
        return fit_results, fit
           
    def _print_fit_results(self, fit_results):
        """
        Prints the fitting results in a formatted manner to summarize key metrics such as resolving power and ratios.

        Args:
            fit_results (dict): A dictionary containing the results and metrics from the fitting process.

        Outputs:
            Prints formatted text summarizing the fit results.
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























    # A class to fit the N2 spectra and return the RP and the 
    # 1st valley over 3rd peak ratio
    
    # instantiate with 
    #   from .base import *
    #   from bessyii.plans.n2fit import N2_fit
    #   N2fit_class = N2_fit(db)
    #   fit_n2 = N2fit_class.fit_n2

    # then use with:
    
    #   fit_n2(identifier,...)