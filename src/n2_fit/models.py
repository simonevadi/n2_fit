import numpy as np
from lmfit.models import LinearModel, SkewedVoigtModel
from lmfit import Parameters

from .helper_functions import find_first_max, find_max_around_theoretical, evaluate_line
from .helper_functions import calculate_skewed_voigt_amplitude, estimate_line_parameters

from .n2_peaks_parameters import theoretical_centers, voigt_intensities

SIGMA_TO_FWHM = 2*np.sqrt(2*np.log(2))
FWHM_TO_SIGMA = 1/SIGMA_TO_FWHM

class N2SkewedVoigtModel:
    """A model class for fitting nitrogen spectra using a skewed Voigt profile with optional background configurations.

    Attributes:
        allowed_backgrounds (list): List of allowed background models for the spectral fitting.
        background (str): Selected background model for fitting.
        fit_gamma (bool): Flag to control whether the gamma parameter of the Voigt profile should be fitted.
    """

    def __init__(self, background='linear', fit_gamma=False):
        """
        Initializes the N2SkewedVoigtModel with specified background and gamma fitting options.

        Args:
            background (str): Type of background model to use. Defaults to 'linear'.
            fit_gamma (bool): Whether to fit the gamma parameter of the Voigt profile. Defaults to False.

        Raises:
            ValueError: If an unsupported background type is specified.
        """
        self.allowed_backgrounds = ['linear', 'offset']
        if background not in self.allowed_backgrounds:
            raise ValueError(f'Allowed values for background are {self.allowed_backgrounds}')
        self.background = background
        self.fit_gamma = fit_gamma


    def config_SkewedVoigtModel(self, model_name, dict_fit,  pars, vary_gamma=False):
        """
        Configures a SkewedVoigtModel with the given parameters.

        Args:
            model_name (str): Name for the model, used as a prefix in parameter naming.
            dict_fit (dict): Dictionary containing the fit parameters and their constraints.
            pars (Parameters): lmfit.Parameters object to which the model parameters are added.
            vary_gamma (bool): Whether to allow the gamma parameter to vary during fitting.

        Returns:
            SkewedVoigtModel: Configured SkewedVoigtModel instance.
        """
        locals()[model_name] = SkewedVoigtModel(prefix=dict_fit['prefix'])   
        return_model_name = locals()[model_name]

        pars.update(getattr(return_model_name,'make_params')())

        pars[''.join((dict_fit['prefix'], 'center'))].set(     value=dict_fit['center'], min=dict_fit['center_low_lim'], max=dict_fit['center_high_lim'] )
        pars[''.join((dict_fit['prefix'], 'sigma'))].set(      value=dict_fit['sigma'],  min=dict_fit['sigma_low_lim'],  max=dict_fit['sigma_high_lim']                     )
        pars[''.join((dict_fit['prefix'], 'amplitude'))].set(  value=dict_fit['amplitude'], min=dict_fit['amplitude_low_lim'], max = dict_fit['amplitude_high_lim']  )
        pars[''.join((dict_fit['prefix'], 'gamma'))].set(      value=dict_fit['gamma'],  vary=vary_gamma, expr='')
        pars[''.join((dict_fit['prefix'], 'skew'))].set(       value=dict_fit['skew_parameter']                                                                  )
        return return_model_name
    
    def make_model(self, dict_fit):
        """
        Constructs a composite model consisting of a background and multiple Voigt profiles based on provided configurations.

        Args:
            dict_fit (dict): Dictionary containing configurations for all components of the model including background and Voigt profiles.

        Returns:
            tuple: A tuple containing the composite model and its associated parameters (lmfit.Model, lmfit.Parameters).
        """
        pars = Parameters()
        
        background = LinearBackground(pars, dict_fit, self.background)
        mod = background.make_model()

        for voigt_n in list(dict_fit.keys()):
            if 'voigt' in voigt_n:
                voigt_dict = dict_fit[voigt_n]
                fit = self.config_SkewedVoigtModel(voigt_n,
                                                voigt_dict,
                                                pars, 
                                                vary_gamma=self.fit_gamma)
                mod = mod + fit
        
        return mod, pars
    
    def prepare_param_for_table(self, fwhm_g:float=0.07, gamma:float=0.0565):
        """
        Prepares and returns fitting parameters for generating a table of spectral data analysis.

        Args:
            fwhm_g (float): Full width at half maximum for Gaussian components.
            gamma (float): Gamma parameter for Lorentzian components.

        Returns:
            dict: A dictionary of fitting parameters for each spectral component.
        """
        sigma_g   = fwhm_g*FWHM_TO_SIGMA
        n_peaks   = 7

        guess = {}
        for index in range(1, n_peaks+1):
            vc           = theoretical_centers[index-1]
            vc_intensity = voigt_intensities[index-1]
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
                'gamma': gamma, 'gamma_low_lim':gamma-1, 'skew_parameter':0
            }
        
        dict_fit['linear'] = {'slope':0, 'intercept':0}
        
        return dict_fit
      
    def get_initial_guess(self, energy, intensity,
                          n_peaks=5, gamma=0.0565, 
                          energy_first_peak='auto'):
        """
        Generates initial guesses for fitting parameters based on provided spectral data.

        Args:
            energy (numpy.array): Array of energy values.
            intensity (numpy.array): Array of intensity values.
            n_peaks (int): Number of peaks to fit.
            gamma (float): Initial gamma value for the fit.
            energy_first_peak (str): Strategy to determine the energy of the first peak, either 'auto' or a specific value.

        Returns:
            dict: A dictionary containing initial guesses for fitting parameters.
        """
        sigma_g     = 0.02
        amp_mf      = 1.5  # scaling value for min/max amplitude
        gamma_min   = 0
        fwhm_g      = sigma_g*SIGMA_TO_FWHM
        sigma_g_min = 0
        sigma_g_max = sigma_g*4
        center_scale_factor = 1
        differences = np.diff(theoretical_centers)
        lin_slope, intercept = estimate_line_parameters(
            [0, energy.shape[0]],  # x-coordinates
            [np.average(intensity[0:3]), np.average(intensity[-3:])]  # y-values
        )
        
        # make sure that the slope is negatinve, else 0
        lin_slope = lin_slope if lin_slope<0 else 0
 
        guess = {}
        for index in range(n_peaks):
            if energy_first_peak == 'auto' and index ==0:
                vc, vc_intensity, argmax = find_first_max(energy,intensity, fwhm_g)
                linear_component = evaluate_line(argmax, lin_slope, intercept)
                vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-linear_component, vc, fwhm_g*4)
                vc_intensity = calculate_skewed_voigt_amplitude(vc, sigma_g, gamma, 0, vc_intensity)
            else:
                linear_component = evaluate_line(argmax, lin_slope, intercept)
                try:
                    vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-linear_component, vc+differences[index-1], fwhm_g*4)
                except ValueError: #
                    raise ValueError('Not enough peaks to fit')
        
                # i decided to use the values from literature instead of the ones I calculate.
                vc_intensity = voigt_intensities[index]
                

            if vc_intensity < 0:
                vc_intensity = guess[f'amp{index-1}']/3

            guess[f'vc{index+1}'] = vc 
            guess[f'amp{index+1}']  = vc_intensity
        
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
                'amplitude_high_lim':np.inf, 
                'gamma': gamma, 'gamma_low_lim':gamma_min, 'skew_parameter':0.0
            }
        
        # guess the slope

        dict_fit['linear'] = {'slope':lin_slope, 'intercept':intercept}
        
        return dict_fit
    

class LinearBackground():
    """Class for handling linear background models within a composite spectral model."""
    def __init__(self, parameters, dict_fit, background_type):
        """
        Initializes the LinearBackground with parameters for constructing a linear model.

        Args:
            parameters (Parameters): lmfit.Parameters object to update with background parameters.
            dict_fit (dict): Dictionary containing initial guesses or fixed values for the background parameters.
            background_type (str): Type of background model, e.g., 'linear' or 'offset'.
        """
        self.parameters = parameters
        self.dict_fit = dict_fit
        self.background_type = background_type
        return 
    
    def make_model(self):
        """
        Constructs a linear background model using the parameters set during initialization.

        Returns:
            LinearModel: An lmfit.LinearModel configured with initial or fixed background parameters.
        """
        lin_mod = LinearModel(prefix='lin_')
        self.parameters.update(lin_mod.make_params())
        if self.background_type == 'linear':
            self.parameters['lin_slope'].set(value=self.dict_fit['linear']['slope'], max=0)
        elif self.background_type == 'offset':
            self.parameters['lin_slope'].set(value=0, vary=False)            
        self.parameters['lin_intercept'].set(value=self.dict_fit['linear']['intercept'])
        
        return lin_mod
    
