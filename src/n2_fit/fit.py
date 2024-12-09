import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pprint import pprint


from .models import N2SkewedVoigtModel, N2SkewedVoigtModelNoLine
from .helper_functions import remove_neg_values
from .helper_functions import extract_bandwidth_and_rp, extract_RP_ratio
from .helper_functions import find_closest_fwhm, format_val, convert_to_json_serializable
from .helper_functions import calculate_rms

from .n2_peaks_parameters import theoretical_centers, theoretical_intensities, voigt_intensities, first_peak

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
        Initializes the N2_fit class with a specified model and database connection.

        Parameters:
            model (str): The name of the model to use for fitting, defaults to 'SkewedVoigt'.
            db (optional): A database connection to retrieve scan data, defaults to None.

        Raises:
            NotImplemented: If a model other than 'SkewedVoigt' is specified.
        """
        self._db = db
        self.tiny = 1.0e-15
        self.s2   = np.sqrt(2)
        self.s2pi = np.sqrt(2*np.pi)

    def _retrieve_spectra(self, identifier, motor=None, detector=None):
        """
        Retrieves spectral data based on the provided identifier, motor, and detector names.

        Parameters:
            identifier (str or int): Identifier for the data set, can be a file path or a database identifier.
            motor (str, optional): Name of the motor to retrieve data from, defaults to the primary motor if not specified.
            detector (str, optional): Name of the detector to retrieve data from, defaults to the primary detector if not specified.

        Returns:
            tuple: Two numpy arrays containing the motor values (x) and the detector values (y).
        """        
        if isinstance(identifier, str) and '.' in identifier:
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
    
    def _fit_n2(self, energy, intensity, dict_fit, fit_gamma=False, print_fit_results=False):
        """
        Performs fitting on N2 spectra using the specified model and parameters.

        Parameters:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of intensity values for the spectral data.
            dict_fit (dict): Dictionary containing fit parameters and model configurations.
            fit_gamma (bool): Flag to indicate whether the gamma parameter should be fitted, defaults to False.
            print_fit_results (bool): Flag to indicate whether to print the fit results, defaults to False.

        Returns:
            ModelResult: The result of the fit containing fit parameters and statistical data.
        """
        norm = np.max(intensity)
        intensity = intensity/norm

        mod, pars = self.model.make_model(dict_fit, fit_gamma=fit_gamma)

            
        out = mod.fit(intensity, pars, x=energy)
        delta = out.eval_uncertainty(x=energy)
        
        if print_fit_results == True:
            print(out.fit_report(min_correl=0.5))

        return out
    
    def plot_initial_guess(self, energy, intensity, mod, pars):
        """
        Plots the initial guess of the fit against the actual data.

        Parameters:
            energy (np.array): Array of energy values for the spectral data.
            intensity (np.array): Array of normalized intensity values for the spectral data.
            mod (Model): The model used for the fit.
            pars (Parameters): Parameters used for the initial guess in the fit.

        Displays:
            A plot showing the initial guess of the fit overlaid on the actual data points.
        """
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


    def analyze_fit_results(self, energy, intensity, fit, n_peaks):
        """
        Analyzes the fitting results to extract key performance metrics such as resolving power and peak ratios.

        Parameters:
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

        fit_results['ratio'] = extract_RP_ratio(energy, intensity, fit)
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
        

    def get_initial_guess(self, scan, n_peaks=5,gamma=0.057, print_initial_guess=False):
        energy, intensity  = self._retrieve_spectra(scan)
        dict_fit = self.model.get_initial_guess(energy,intensity,
                                                theoretical_centers,
                                                theoretical_intensities,
                                                n_peaks, gamma=gamma)
        if print_initial_guess:
            pprint(dict_fit)

        return dict_fit

    def fit_n2(self, scan, dict_fit=None, n_peaks=5, 
               plot_initial_guess=False, print_fit_results=False, 
               save_results=False, show_results=True, fwhm_l:float=114, 
               model:str='SkewedVoigt', motor=None, detector=None):
        """
        Orchestrates the fitting process for N2 spectral data, including retrieving data, performing the fit,
        analyzing results, and plotting.

        Parameters:
            scan (str): Path to the scan file or database identifier.
            dict_fit (dict, optional): Predefined fitting parameters. If None, defaults are used.
            n_peaks (int): Number of peaks to fit.
            plot_initial_guess (bool): If True, plots the initial guesses.
            print_fit_results (bool): If True, prints the fit report.
            save_results (bool or str): If True or a path is provided, saves the fit results to a JSON file.
            show_results (bool): If True, displays the plot.
            gamma (float): Gamma value for fitting, defaults to 0.057.

        Effects:
            Performs the fit, saves results, and optionally displays plots.
        """
        if model == 'SkewedVoigt':
            self.model = N2SkewedVoigtModel()
        else:
            raise NotImplemented('No other Model is Implemented')
        

        print(f'Starting the fit for {scan}')
        gamma = fwhm_l/2000
        if save_results:
            if not os.path.exists(save_results):
                os.makedirs(save_results)
        energy, intensity  = self._retrieve_spectra(scan, motor=motor, detector=detector)
        if dict_fit is None:
            dict_fit = self.model.get_initial_guess(energy,intensity, theoretical_centers,
                                                    theoretical_intensities, 
                                                    n_peaks, gamma=gamma)
        
        model, parameters = self.model.make_model(dict_fit)
        
        if plot_initial_guess:
            self.plot_initial_guess(energy, intensity, model, parameters)

        out = self._fit_n2(energy,intensity, dict_fit, print_fit_results=print_fit_results)
        
        fit_results = self.analyze_fit_results(energy, 
                                                intensity, 
                                                out,
                                                n_peaks=n_peaks)
        if isinstance(scan, int):
            title = f'scan_{scan}'
        else:
            base_name = os.path.basename(scan)
            title = os.path.splitext(base_name)[0]
        if save_results:
            json_ready_results = convert_to_json_serializable(fit_results)
            analysis_save_path = os.path.join(save_results, f'{title}.json')
            with open(analysis_save_path, 'w') as json_file:
                json.dump(json_ready_results, json_file, indent=4)

        self._print_fit_results(fit_results)
        self.plot_fit(title, energy, intensity, out, fit_results, save_results=save_results, show_results=show_results)

    def fit_n2_3peaks(self, scan, dict_fit=None, n_peaks=5, 
               plot_initial_guess=False, print_fit_results=False, 
               save_results=False, show_results=True, fwhm_l:float=114, 
               model:str='SkewedVoigt', motor=None, detector=None):
        """
        Orchestrates the fitting process for N2 spectral data, including retrieving data, performing the fit,
        analyzing results, and plotting.

        Parameters:
            scan (str): Path to the scan file or database identifier.
            dict_fit (dict, optional): Predefined fitting parameters. If None, defaults are used.
            n_peaks (int): Number of peaks to fit.
            plot_initial_guess (bool): If True, plots the initial guesses.
            print_fit_results (bool): If True, prints the fit report.
            save_results (bool or str): If True or a path is provided, saves the fit results to a JSON file.
            show_results (bool): If True, displays the plot.
            gamma (float): Gamma value for fitting, defaults to 0.057.

        Effects:
            Performs the fit, saves results, and optionally displays plots.
        """

        if model == 'SkewedVoigt':
            self.model = N2SkewedVoigtModelNoLine()
        else:
            raise NotImplemented('No other Model is Implemented')
        
        print(f'Starting the fit for {scan}')
        gamma = fwhm_l/2000
        if save_results:
            if not os.path.exists(save_results):
                os.makedirs(save_results)
        energy, intensity  = self._retrieve_spectra(scan, motor=motor, detector=detector)
        intensity = intensity-np.min(intensity)
        if dict_fit is None:
            dict_fit = self.model.get_initial_guess(energy,intensity, theoretical_centers,
                                                    theoretical_intensities, 
                                                    n_peaks, gamma=gamma)
        
        model, parameters = self.model.make_model(dict_fit)
        
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
        Prints the fitting results in a formatted manner to summarize the key metrics such as resolving power and ratios.

        Parameters:
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