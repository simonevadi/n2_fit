import numpy as np
from lmfit.models import LinearModel, SkewedVoigtModel
from lmfit import Parameters

from .helper_functions import find_first_max, find_max_around_theoretical, evaluate_line
from .helper_functions import calculate_skewed_voigt_amplitude, estimate_line_parameters

class N2SkewedVoigtModel:

    def config_SkewedVoigtModel(self, model_name, prefix_, value_center, value_center_min, 
                                value_center_max, value_sigma, value_sigma_min, 
                                value_sigma_max, value_amp, value_amp_min, value_amp_max, value_gamma, 
                                value_skew, pars, vary_gamma=False):
        """
        Configure a SkewdVoigtModel to be used in the fit when fix_parameters = False

        Parameters
        ----------
        model_name : string
             the name of the model
        prefix_    : string
             the name of the skewdvoigt peak
        value_center: float
             the center of the peak
        value_center_min: float
             the lower bound for the center of the peak for the fit routine
        value_center_max: float
             the upper bound for the center of the peak for the fit routine
        value_sigma: float
             the sigma of the gaussian component of the voigt peak
        value_sigma_min: float
             the lower bound for the sigma for the fit routine
        value_sigma_max: float
             the upper bound for the sigma for the fit routine
        value_amp: float
             the value for the amplitude of the peak
        value_amp_min: float
             the lower bound for the amplitude for the fit routine
        value_gamma: float
             the gamma value for the loretzian component of the voigt peak. This parameter is not fitted.
        value_skew: float
             the skew parameter for the voigt peak (defines peak asimmetry)
        pars: lmfit parameter class of a model


        Return
        --------
        x,y : np.array
            two arrays without negative values
        """
        value_amp_min = value_amp_min if value_amp_min>=0 else 0
        value_amp = value_amp if value_amp>=0 else 0
        locals()[model_name] = SkewedVoigtModel(prefix=prefix_)   
        return_model_name = locals()[model_name]

        pars.update(getattr(return_model_name,'make_params')())

        pars[''.join((prefix_, 'center'))].set(     value=value_center, min=value_center_min, max=value_center_max                    )
        pars[''.join((prefix_, 'sigma'))].set(      value=value_sigma,  min=value_sigma_min,  max=value_sigma_max                     )
        pars[''.join((prefix_, 'amplitude'))].set(  value=value_amp, min=value_amp_min, max = value_amp_max  )
        pars[''.join((prefix_, 'gamma'))].set(      value=value_gamma,  vary=vary_gamma, expr='')
        pars[''.join((prefix_, 'skew'))].set(       value=value_skew                                                                  )
        return return_model_name
    
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
                fit = self.config_SkewedVoigtModel(voigt_n,
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
    
    def prepare_param_for_table(self, theoretical_centers:np.array, voigt_intensities:np.array, fwhm_g:float=0.07, gamma:float=0.0565):

        sigma_g   = fwhm_g/2.35
        fwhm_g    = 2.355*sigma_g
        intercept = 0
        n_peaks   = 7
        lin_slope = 0
        skew_param = 0
        
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
                'gamma': gamma, 'gamma_low_lim':gamma-1, 'skew_parameter':skew_param
            }
        
        dict_fit['linear'] = {'slope':lin_slope, 'intercept':intercept}
        
        return dict_fit
    
    def get_initial_guess(self,energy,intensity,theoretical_centers, theoretical_intensities, n_peaks=5, gamma=0.0565, energy_first_peak='auto'):
        # normalize intensity
        norm = np.max(intensity)
        intensity = intensity/norm
        sigma_g     = 0.01
        amp_mf      = 1.3  #scaling value for minimal amplitude
        gamma_min   = 0
        fwhm_g      = 2.355*sigma_g
        sigma_g_min = 0
        sigma_g_max = np.inf
        center_scale_factor = 2
        differences = np.diff(theoretical_centers)
        lin_slope, intercept = estimate_line_parameters(
            [0, energy.shape[0]],  # x-coordinates
            [np.average(intensity[0:3]), np.average(intensity[-3:])]  # y-values
        )
        guess = {}
        for index in range(1, n_peaks+1):
            if energy_first_peak == 'auto' and index ==1:
                vc, vc_intensity, argmax = find_first_max(energy,intensity, fwhm_g)
                linear_component = evaluate_line(argmax, lin_slope, intercept)
                vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-linear_component, vc, fwhm_g*4)
                vc_intensity = calculate_skewed_voigt_amplitude(vc, sigma_g, gamma, 0, vc_intensity)
            else:
                linear_component = evaluate_line(argmax, lin_slope, intercept)
                vc, vc_intensity, argmax = find_max_around_theoretical(energy, intensity-linear_component, vc+differences[index-1], fwhm_g*4)
                vc_intensity = calculate_skewed_voigt_amplitude(vc, sigma_g, gamma, 0, vc_intensity)

            if vc_intensity < 0:
                vc_intensity = guess[f'amp{index-1}']/3

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